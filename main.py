import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from datetime import datetime, timedelta
from collections import deque
import os

# Configure page 配置页面 
st.set_page_config(page_title="Stillstandserkennung", layout="wide")

# Model path 模型路径
MODEL_PATH = "best.pt"

# Class name mapping 类别名称映射
CLASS_NAMES = {
    0: "Werkzeug",
    1: "Aufnahme",
    2: "geschlossen",
    3: "Haken",
    4: "Werkzeug justieren",
    5: "Druckluftpistole",
    6: "Handschuh"
}

# Machine status constants 机器状态常量
STATUS = {
    "WORKING": "Arbeit",
    "WAITING_DOOR_CLOSED": "Stillstand-Tür geschlossen-Warten",
    "WAITING_DOOR_OPEN": "Stillstand-Tür geöffnet-Warten",
    "HAKEN_ENTFERNEN": "Stillstand-Späne von Haken entfernen",
    "WERKZEUG_JUSTIEREN": "Stillstand-Werkzeug justieren",
    "DRUCKLUFT_ENTFERNEN": "Stillstand-Späne von Druckluftpistole entfernen",
    "HAND_ENTFERNEN": "Stillstand-Späne von Hand entfernen",
    "VERFAHRBEWEGUNG": "Stillstand-Verfahrbewegung"
}

@st.cache_resource
def load_model():
    """Load YOLOv8 model (cached) YOLOv8模型加载（缓存）"""
    return YOLO(MODEL_PATH)

def calculate_frame_diff(roi1, roi2):
    """Calculate the difference between two frames 计算两帧之间的差异"""
    if roi1 is None or roi2 is None or roi1.size == 0 or roi2.size == 0:
        return 0
    
    # Ensure the two ROIs are the same size 确保两个ROI大小相同
    if roi1.shape != roi2.shape:
        roi2 = cv2.resize(roi2, (roi1.shape[1], roi1.shape[0]))
    
    # Convert to grayscale 转换为灰度图
    if len(roi1.shape) == 3:
        gray_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi1 = roi1
    
    if len(roi2.shape) == 3:
        gray_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi2 = roi2
    
    # Calculate the difference 计算差异
    diff = cv2.absdiff(gray_roi1, gray_roi2)
    return np.mean(diff)

def calculate_brightness_variance(roi):
    """Calculate the average brightness variance of the region (optimized version) 计算区域平均亮度方差（优化版本）"""
    if roi is None or roi.size == 0:
        return 0
        
    # Convert to grayscale 转换为灰度图
    if len(roi.shape) == 3:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = roi
    
    # Directly calculate the overall variance 直接计算整体方差
    return np.var(gray_roi)

def get_bbox_roi(frame, bbox):
    """Get the region of interest from the bounding box 从边界框获取感兴趣区域"""
    if frame is None or bbox is None:
        return None
    
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Ensure the boundaries are within the image range 确保边界在图像范围内
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return frame[y1:y2, x1:x2]

def get_area_between_objects(frame, werkzeug_bbox, aufnahme_bbox):
    """Calculate the ROI of the area between two objects 计算两个对象之间区域的ROI"""
    if frame is None or werkzeug_bbox is None or aufnahme_bbox is None:
        return None
    
    height, width = frame.shape[:2]
    
    # Extract coordinates from the bounding boxes 从边界框中提取坐标
    wx1, wy1, wx2, wy2 = werkzeug_bbox
    ax1, ay1, ax2, ay2 = aufnahme_bbox
    
    # Define the cabin area as from the bottom right of Aufnahme to the top left of Werkzeug 舱内区域定义为Aufnahme右下角到Werkzeug左上角
    left = ax1  # Aufnahme左边界
    top = wy1   # Werkzeug上边界
    right = ax2 # Aufnahme右边界
    bottom = ay2 # Aufnahme下边界
    
    # Ensure the boundaries are within the image range 确保边界在图像范围内
    left = max(0, int(left))
    top = max(0, int(top))
    right = min(width, int(right))
    bottom = min(height, int(bottom))
    
    if right <= left or bottom <= top:
        return None
    
    return frame[top:bottom, left:right]

def analyze_frame(model, frame, prev_frames, config):
    """Analyze video frame and determine machine status 分析视频帧并确定机器状态"""
    # Use more efficient inference parameters 使用更高效的推理参数
    results = model(frame, conf=config["confidence"], verbose=False)
    result = results[0]
    
    # Extract detected classes and bounding boxes 提取检测到的类别和边界框
    boxes = result.boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidence_scores = boxes.conf.cpu().numpy()
    xyxy_boxes = boxes.xyxy.cpu().numpy()
    
    # Create a copy of the frame for visualization 创建用于可视化的帧副本
    vis_frame = frame.copy()
    
    # Check if "geschlossen" (class 2) is detected 检查是否检测到"geschlossen"(类别2)
    is_door_closed = 2 in class_ids
    
    # Initialize bounding boxes for each class 初始化各类别边界框
    werkzeug_bbox = None
    aufnahme_bbox = None
    werkzeug_conf = 0
    aufnahme_conf = 0
    
    # Extract bounding boxes for each class 提取各类别的边界框
    detected_classes = {}
    for i, class_id in enumerate(class_ids):
        box = xyxy_boxes[i]
        conf = confidence_scores[i]
        
        # Store detected classes 存储检测到的类别
        detected_classes[class_id] = (box, conf)
        
        # Specifically store Werkzeug and Aufnahme bounding boxes 特别存储Werkzeug和Aufnahme的边界框
        if class_id == 0 and conf > werkzeug_conf:  # Werkzeug
            werkzeug_bbox = box
            werkzeug_conf = conf
        elif class_id == 1 and conf > aufnahme_conf:  # Aufnahme
            aufnahme_bbox = box
            aufnahme_conf = conf
        
        # Draw bounding boxes and labels 绘制边界框和标签
        color = (0, 255, 0) if class_id == 2 else (255, 0, 0)  # 为geschlossen使用绿色，其他使用红色
        cv2.rectangle(vis_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        label = f"{CLASS_NAMES.get(class_id, 'Unbekannt')}: {conf:.2f}"
        cv2.putText(vis_frame, label, (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Get current ROIs 获取当前ROI
    werkzeug_roi = get_bbox_roi(frame, werkzeug_bbox) if werkzeug_bbox is not None else None
    aufnahme_roi = get_bbox_roi(frame, aufnahme_bbox) if aufnahme_bbox is not None else None
    between_roi = get_area_between_objects(frame, werkzeug_bbox, aufnahme_bbox) if werkzeug_bbox is not None and aufnahme_bbox is not None else None
    
    # Initialize measurement results 初始化测量结果
    werkzeug_movement = 0
    aufnahme_change = 0  
    between_movement = 0
    verfahrbewegung_detected = False
    
    # Calculate Werkzeug position change (using bounding box center point coordinates) 计算Werkzeug位置变化（使用边界框中心点坐标）
    if werkzeug_bbox is not None and "prev_werkzeug_position" in prev_frames:
        # Calculate current bounding box center point 计算当前边界框中心点
        curr_x = (werkzeug_bbox[0] + werkzeug_bbox[2]) / 2
        curr_y = (werkzeug_bbox[1] + werkzeug_bbox[3]) / 2
        curr_position = (curr_x, curr_y)
        
        # Get previous position 获取之前的位置
        prev_position = prev_frames["prev_werkzeug_position"]
        
        # Calculate position change (Euclidean distance) 计算位置变化（欧氏距离）
        werkzeug_movement = np.sqrt((curr_position[0] - prev_position[0])**2 + 
                                  (curr_position[1] - prev_position[1])**2)
        
        # Update position history 更新位置历史
        prev_frames["prev_werkzeug_position"] = curr_position
        
        # Display Werkzeug movement value and position on the image 在图像上显示Werkzeug移动值和位置
        cv2.circle(vis_frame, (int(curr_x), int(curr_y)), 5, (0, 255, 255), -1)  # 绘制中心点
    elif werkzeug_bbox is not None:
        # Initialize position history 初始化位置历史
        curr_x = (werkzeug_bbox[0] + werkzeug_bbox[2]) / 2
        curr_y = (werkzeug_bbox[1] + werkzeug_bbox[3]) / 2
        prev_frames["prev_werkzeug_position"] = (curr_x, curr_y)
        
        # Display the initial position on the image 在图像上显示初始位置
        cv2.putText(vis_frame, "Werkzeug Position initialisiert", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.circle(vis_frame, (int(curr_x), int(curr_y)), 5, (0, 255, 255), -1)  # 绘制中心点
    
    # Calculate Aufnahme brightness change 计算Aufnahme的亮度变化
    if len(prev_frames["aufnahme"]) > 0 and aufnahme_roi is not None:
        # Calculate Aufnahme change (using brightness variance) 计算Aufnahme的变化（使用亮度方差）
        curr_brightness = calculate_brightness_variance(aufnahme_roi)
        prev_brightness = calculate_brightness_variance(prev_frames["aufnahme"][-1])
        aufnahme_change = abs(curr_brightness - prev_brightness) * config["aufnahme_change_factor"]
    
    # Calculate cabin area change 计算舱内区域变化
    if len(prev_frames["between"]) > 0 and between_roi is not None:
        # Calculate movement between areas 计算区域之间的移动
        prev_between_roi = prev_frames["between"][-1]
        between_movement = calculate_frame_diff(between_roi, prev_between_roi)
        
        # Use the difference between before and after frames in the cabin to detect Verfahrbewegung 使用前后两帧的舱内差异来检测Verfahrbewegung
        if not is_door_closed:
            # If the frame difference is greater than the threshold and no specific tool classes (3,4,5,6) are detected, then it is determined to be Verfahrbewegung 如果帧差异大于阈值且未检测到特定工具类别(3,4,5,6)，则判定为Verfahrbewegung
            special_tools_detected = any(tool_id in detected_classes for tool_id in [3, 4, 5, 6])
            verfahrbewegung_detected = (between_movement > config["between_movement_threshold"]) and not special_tools_detected
    
    # Update ROI history 更新ROI历史
    if werkzeug_roi is not None:
        prev_frames["werkzeug"].append(werkzeug_roi)
    if aufnahme_roi is not None:
        prev_frames["aufnahme"].append(aufnahme_roi)
    if between_roi is not None:
        prev_frames["between"].append(between_roi)
    
    # Determine machine status 确定机器状态
    machine_status = determine_machine_status(
        is_door_closed, 
        detected_classes, 
        werkzeug_movement, 
        aufnahme_change,
        between_movement,
        verfahrbewegung_detected,
        config
    )
    
    return vis_frame, machine_status, werkzeug_movement, aufnahme_change, between_movement, is_door_closed, detected_classes, verfahrbewegung_detected

def determine_machine_status(is_door_closed, detected_classes, werkzeug_movement, aufnahme_change, between_movement, verfahrbewegung_detected, config):
    """Determine machine status 确定机器状态"""
    # If the door is closed 如果门关闭
    if is_door_closed:
        # Check Werkzeug movement 检查Werkzeug移动
        if werkzeug_movement > config["werkzeug_movement_threshold"]:
            return STATUS["WORKING"]
        
        # Check Aufnahme change 检查Aufnahme变化
        if aufnahme_change > config["aufnahme_change_threshold"]:
            return STATUS["WORKING"]
        
        # If the door is closed but no movement or rotation is detected 如果门关闭但没有检测到移动或旋转
        return STATUS["WAITING_DOOR_CLOSED"]
    else:
        # Door is open 门开启
        # Check if specific classes exist 检查是否存在特定类别
        if 3 in detected_classes:  # Haken
            return STATUS["HAKEN_ENTFERNEN"]
        elif 4 in detected_classes:  # Werkzeug justieren
            return STATUS["WERKZEUG_JUSTIEREN"]
        elif 5 in detected_classes:  # Druckluftpistole
            return STATUS["DRUCKLUFT_ENTFERNEN"]
        elif 6 in detected_classes:  # Handschuh
            return STATUS["HAND_ENTFERNEN"]
        # Check verfahrbewegung_detected status 检查verfahrbewegung_detected状态
        elif verfahrbewegung_detected:
            return STATUS["VERFAHRBEWEGUNG"]
        else:
            return STATUS["WAITING_DOOR_OPEN"]

def format_time(seconds):
    """Format seconds to minutes:seconds format 将秒数格式化为分:秒格式"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def main():
    st.title("Zerspanungsmaschine-Stillstandserkennung")
    
    # Initialize session state 初始化会话状态
    if 'has_analyzed' not in st.session_state:
        st.session_state.has_analyzed = False
    if 'history_df' not in st.session_state:
        st.session_state.history_df = None
    if 'status_periods_df' not in st.session_state:
        st.session_state.status_periods_df = None
    if 'video_duration' not in st.session_state:
        st.session_state.video_duration = 0
    if 'verfugbarkeit' not in st.session_state:
        st.session_state.verfugbarkeit = 0
    if 'actual_production_time_sec' not in st.session_state:
        st.session_state.actual_production_time_sec = 0
    if 'unplanned_downtime_sec' not in st.session_state:
        st.session_state.unplanned_downtime_sec = 0
    if 'planned_downtime_sec' not in st.session_state:
        st.session_state.planned_downtime_sec = 0
    if 'status_history' not in st.session_state:
        st.session_state.status_history = []
    if 'timestamps' not in st.session_state:
        st.session_state.timestamps = []
    if 'category_duration' not in st.session_state:
        st.session_state.category_duration = {}
    if 'video_timestamps' not in st.session_state:  # Store real video time 存储真实视频时间
        st.session_state.video_timestamps = []
    
    # Sidebar configuration 侧边栏配置
    st.sidebar.header("Konfigurationsparameter")
    
    confidence = st.sidebar.slider("Konfidenz-Schwellenwert", 0.1, 1.0, 0.5, 0.05, key="confidence")
    werkzeug_movement_threshold = st.sidebar.slider("Werkzeug-Bewegungsschwelle", 0.1, 2.0, 1.0, 0.1, key="werkzeug_threshold")
    aufnahme_change_threshold = st.sidebar.slider("Aufnahme-Helligkeitsänderungsschwelle", 1.0, 20.0, 10.0, 1.0, key="aufnahme_threshold")
    aufnahme_change_factor = st.sidebar.slider("Aufnahme-Helligkeitsverstärkungsfaktor", 0.1, 10.0, 1.0, 0.1, key="aufnahme_factor")
    between_movement_threshold = st.sidebar.slider("Innenraum-Frame-Differenzschwelle", 0.1, 10.0, 5.0, 0.5, key="between_threshold")
    process_every_n_frames = st.sidebar.slider("Jeder N-te Frame verarbeiten", 1, 20, 10, 1, key="frame_process")
    state_change_frames = st.sidebar.slider("Erforderliche aufeinanderfolgende Frames für Statusänderung", 1, 10, 3, 1, key="state_frames")
    
    config = {
        "confidence": confidence,
        "werkzeug_movement_threshold": werkzeug_movement_threshold,
        "aufnahme_change_threshold": aufnahme_change_threshold,
        "aufnahme_change_factor": aufnahme_change_factor,
        "between_movement_threshold": between_movement_threshold,
        "process_every_n_frames": process_every_n_frames,
        "state_change_frames": state_change_frames
    }
    
    # Create tabs 创建选项卡
    tab1, tab2, tab3 = st.tabs(["Analyse", "Historische Daten", "Verfügbarkeitsanalyse"])
    
    # Check if reanalysis is needed or to display existing results 检查是否需要重新分析或显示现有结果
    should_analyze = not st.session_state.has_analyzed
    
    with tab1:
        # Create a two-column layout 创建一个两列布局
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Videoanalyse")
            source = st.selectbox(
                "Videoquelle auswählen",
                ["Videodatei hochladen", "Kamera"]
            )
            
            if source == "Videodatei hochladen":
                uploaded_file = st.file_uploader("Videodatei hochladen", type=["mp4", "avi", "mov"])
                if uploaded_file is not None:
                    # Save the uploaded file 保存上传的文件
                    with open("temp_video.mp4", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    video_path = "temp_video.mp4"
                    should_analyze = True  # Reanalyze when a new file is uploaded 新上传文件时重新分析
                    st.session_state.has_analyzed = False
                else:
                    st.info("Bitte laden Sie eine Videodatei hoch")
                    should_analyze = False
                    video_path = None
            else:
                video_path = 0  # Use default camera 使用默认摄像头
                should_analyze = True  # Always analyze when using camera 使用摄像头时始终分析
                st.session_state.has_analyzed = False
            
            # Add reanalysis button 添加重新分析按钮
            if st.session_state.has_analyzed and st.button("Neue Analyse starten"):
                should_analyze = True
                st.session_state.has_analyzed = False
            
            # Create video display location 创建视频显示位置
            video_placeholder = st.empty()
        
        with col2:
            #st.header("Status")
            # Create status display location 创建状态显示位置
            status_placeholder = st.empty()
            metrics_placeholder = st.empty()
        
        if should_analyze and video_path is not None:
            # Load model 加载模型
            with st.spinner("Lade Modell..."):
                model = load_model()
            
            # Create data structures to store historical information 创建数据结构来存储历史信息
            status_history = []
            timestamps = []
            video_timestamps = []  # Store real video timestamps 存储真实视频时间戳
            werkzeug_movements = []
            aufnahme_changes = []
            between_movements = []
            door_states = []
            detected_classes_history = [] 
            
            # Create ROI history queue 创建ROI历史队列
            prev_frames = {
                "werkzeug": deque(maxlen=5),
                "aufnahme": deque(maxlen=5),
                "between": deque(maxlen=5)
            }
            
            # Status change counter 状态变化计数器
            status_counter = {}
            current_status = None
            
            # Start processing video 开始处理视频
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                st.error("Die Videoquelle kann nicht geöffnet werden")
                return
            
            # Get video information 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if video_path != 0 else float('inf')
            
            # Create stop button 创建停止按钮
            stop_button = st.button("Verarbeitung beenden")
            
            # Create progress bar (only for video files) 创建进度条（仅用于视频文件）
            if video_path != 0:
                progress_bar = st.progress(0)
            
            frame_idx = 0
            start_time = datetime.now()
            
            try:
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # Calculate the actual video time of the current frame (for all frames) 计算当前帧的实际视频时间（针对所有帧）
                    video_time_seconds = frame_idx / fps
                    
                    # Process every N frames 每N帧处理一次
                    if frame_idx % config["process_every_n_frames"] == 0:
                        try:
                            # Process frame 处理帧
                            vis_frame, status, werkzeug_movement, aufnahme_change, between_movement, is_door_closed, detected_classes, verfahrbewegung_detected = analyze_frame(
                                model, frame, prev_frames, config
                            )
                            
                            # Status change processing 状态变化处理
                            if status != current_status:
                                if status not in status_counter:
                                    status_counter[status] = 0
                                status_counter[status] += 1
                                
                                # If consecutive frames reach the threshold, change the status 如果连续帧达到阈值，则更改状态
                                if status_counter[status] >= config["state_change_frames"]:
                                    current_status = status
                                    # Reset counter 重置计数器
                                    status_counter = {k: 0 for k in status_counter}
                            else:
                                # Reset counters for other statuses 重置其他状态的计数器
                                status_counter = {k: 0 for k in status_counter if k != status}
                                if status not in status_counter:
                                    status_counter[status] = 0
                                status_counter[status] = min(status_counter[status] + 1, config["state_change_frames"])
                            
                            # Record historical data 记录历史数据
                            if video_path != 0:
                                # Use time relative to the start of the video 使用相对于视频开始的时间
                                timestamp = start_time + timedelta(seconds=frame_idx/fps)
                            else:
                                # Use current system time 使用当前系统时间
                                timestamp = datetime.now()
                                
                            timestamps.append(timestamp)
                            video_timestamps.append(video_time_seconds)  # Add real video time 添加真实视频时间
                            status_history.append(current_status if current_status else status)
                            werkzeug_movements.append(werkzeug_movement)
                            aufnahme_changes.append(aufnahme_change)
                            between_movements.append(between_movement)
                            door_states.append("Tür geschlossen" if is_door_closed else "Tür geöffnet")
                            detected_classes_history.append(detected_classes) 
                            
                            # Convert format for Streamlit display 转换格式以便Streamlit显示
                            vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(vis_frame_rgb, caption="Analyse", width=True)
                            
                            # Create real-time status metrics panel 创建实时状态指标面板
                            with status_placeholder.container():
                                st.subheader("Aktueller Maschinenstatus")
                                current_display_status = current_status if current_status else status
                                status_color = "red" if current_display_status != STATUS["WORKING"] else "green"
                                st.markdown(f"<h3 style='color: {status_color};'>{current_display_status}</h3>", unsafe_allow_html=True)
                            
                            # Display real-time measurement values in the sidebar 在侧边栏显示实时测量值
                            with metrics_placeholder.container():
                                st.subheader("Parameter")
                                col1, col2 = st.columns(2)
                                
                                col1.metric("Türstatus", "Geschlossen" if is_door_closed else "Geöffnet")
                                col2.metric("Werkzeug-Bewegung", f"{werkzeug_movement:.2f}")
                                
                                col1.metric("Aufnahme-Helligkeitsänderung", f"{aufnahme_change:.2f}")
                                col2.metric("Innenraum-Differenz", f"{between_movement:.2f}")
                                
                                st.subheader("Objekt")
                                tool_col1, tool_col2 = st.columns(2)
                                
                                # Check if specific classes exist 检查是否存在特定类别
                                haken_present = 3 in detected_classes
                                werkzeug_just_present = 4 in detected_classes
                                druckluft_present = 5 in detected_classes
                                handschuh_present = 6 in detected_classes
                                
                                # Display detection status 显示检测状态
                                tool_col1.metric("Haken", "Ja" if haken_present else "Nein")
                                tool_col1.metric("Werkzeug justieren", "Ja" if werkzeug_just_present else "Nein")
                                
                                tool_col2.metric("Druckluftpistole", "Ja" if druckluft_present else "Nein")
                                tool_col2.metric("Handschuh", "Ja" if handschuh_present else "Nein")
                                
                                # Display video time 显示视频时间
                                if video_path != 0:
                                    formatted_video_time = format_time(video_time_seconds)
                                    st.metric("Videozeit", formatted_video_time)
                            
                            # Update progress bar (only for video files) 更新进度条（仅用于视频文件）
                            if video_path != 0 and frame_count > 0:
                                progress_bar.progress(min(frame_idx / frame_count, 1.0))
                        
                        except Exception as e:
                            st.error(f"Fehler bei der Frame-Verarbeitung: {e}")
                            break
                    
                    frame_idx += 1
                    
                    # Use time.sleep to simulate real-time processing 使用time.sleep来模拟实时处理
                    if video_path != 0:  # Delay control only for video files 仅对视频文件进行延迟控制
                        time.sleep(1/(fps*2))  # Control processing speed, slightly faster than real-time 控制处理速度，略快于实时
                    
                    # Check if there is a stop request 检查是否有停止请求
                    if stop_button:
                        break
            
            except Exception as e:
                st.error(f"Video-Verarbeitungsfehler: {e}")
            finally:
                cap.release()
            
            # Display results after processing is complete 处理完成后显示结果
            if len(status_history) > 0:
                st.session_state.has_analyzed = True
                
                # Skip the first few frames of data (may be unstable) 跳过前几帧数据（可能不稳定）
                skip_frames = min(10, len(status_history) // 10)
                
                if len(status_history) > skip_frames:
                    timestamps = timestamps[skip_frames:]
                    video_timestamps = video_timestamps[skip_frames:]  # Also skip video timestamps 也跳过视频时间戳
                    status_history = status_history[skip_frames:]
                    werkzeug_movements = werkzeug_movements[skip_frames:]
                    aufnahme_changes = aufnahme_changes[skip_frames:]
                    between_movements = between_movements[skip_frames:]
                    door_states = door_states[skip_frames:]
                    detected_classes_history = detected_classes_history[skip_frames:]
                
                # Save total video duration 保存视频总时长
                if video_path != 0:
                    st.session_state.video_duration = frame_count / fps
                
                # Calculate total time and Verfügbarkeit 计算总时间和Verfügbarkeit
                total_time = len(status_history)
                
                # Calculate production time and different types of downtime 计算生产时间和不同类型的停机时间
                actual_production_time = sum(1 for status in status_history if status == STATUS["WORKING"])
                
                # Calculate unplanned downtime (all waiting states, except VERFAHRBEWEGUNG) 计算未计划停机时间 (所有等待状态，除了VERFAHRBEWEGUNG)
                unplanned_downtime = sum(1 for status in status_history if status != STATUS["WORKING"] and status != STATUS["VERFAHRBEWEGUNG"])
                
                # Calculate planned downtime (VERFAHRBEWEGUNG) 计算计划停机时间 (VERFAHRBEWEGUNG)
                planned_downtime = sum(1 for status in status_history if status == STATUS["VERFAHRBEWEGUNG"])
                
                # Calculate Verfügbarkeit 计算Verfügbarkeit
                if (actual_production_time + unplanned_downtime) > 0:
                    verfugbarkeit = (actual_production_time / (actual_production_time + unplanned_downtime)) * 100
                else:
                    verfugbarkeit = 0
                
                # Calculate actual time for each category (considering process_every_n_frames) 计算各类别的实际时间（考虑process_every_n_frames）
                actual_frames = actual_production_time * config["process_every_n_frames"]
                unplanned_frames = unplanned_downtime * config["process_every_n_frames"]
                planned_frames = planned_downtime * config["process_every_n_frames"]
                
                # Convert to seconds 转换为秒
                actual_production_time_sec = actual_frames / fps
                unplanned_downtime_sec = unplanned_frames / fps
                planned_downtime_sec = planned_frames / fps
                
                # Save to session state 保存到会话状态
                st.session_state.verfugbarkeit = verfugbarkeit
                st.session_state.actual_production_time_sec = actual_production_time_sec
                st.session_state.unplanned_downtime_sec = unplanned_downtime_sec
                st.session_state.planned_downtime_sec = planned_downtime_sec
                st.session_state.status_history = status_history
                st.session_state.timestamps = timestamps
                st.session_state.video_timestamps = video_timestamps  # Save real video time 保存真实视频时间
                
                st.success(f"Videoverarbeitung abgeschlossen! Verfügbarkeit: {verfugbarkeit:.2f}%")
                
                # Create tool detection status for each frame 为每一帧创建工具检测状态
                haken_detected = []
                druckluft_detected = []
                handschuh_detected = []
                werkzeug_just_detected = []
                
                # Extract tool detection status from recorded detection results 从记录的检测结果中提取工具检测状态
                for classes_dict in detected_classes_history:
                    haken_detected.append(3 in classes_dict)
                    druckluft_detected.append(5 in classes_dict)
                    handschuh_detected.append(6 in classes_dict)
                    werkzeug_just_detected.append(4 in classes_dict)
                
                # Format video timestamps 格式化视频时间戳
                formatted_video_timestamps = [format_time(t) for t in video_timestamps]
                
                # Create historical data DataFrame, using real video time 创建历史数据DataFrame，使用真实视频时间
                history_df = pd.DataFrame({
                    "Zeitstempel": formatted_video_timestamps,  # Use formatted video timestamps 使用格式化的视频时间戳
                    "Videozeit (Sekunden)": video_timestamps,  # Add original seconds for sorting 添加原始秒数用于排序
                    "Maschinenstatus": status_history,
                    "Werkzeug-Positionsänderung": werkzeug_movements,
                    "Aufnahme-Helligkeitsänderung": aufnahme_changes,
                    "Innenraum-Frame-Differenz": between_movements,
                    "Türstatus": door_states,
                    "Haken": haken_detected,
                    "Druckluftpistole": druckluft_detected,
                    "Handschuh": handschuh_detected,
                    "Werkzeug justieren": werkzeug_just_detected
                })
                
                # Save DataFrame to session state 将DataFrame保存到会话状态
                st.session_state.history_df = history_df
                
                # Create status duration table (using actual video time) 创建状态持续时间表格（使用实际视频时间）
                status_periods = []
                if len(status_history) > 0:
                    current_status = status_history[0]
                    start_idx = 0
                    
                    for i in range(1, len(status_history)):
                        if status_history[i] != current_status:
                            # Status change, record the duration of the previous status 状态变化，记录前一个状态的持续时间
                            duration = video_timestamps[i] - video_timestamps[start_idx]
                            status_periods.append({
                                "Maschinenstatus": current_status,
                                "Startzeit": formatted_video_timestamps[start_idx],
                                "Endzeit": formatted_video_timestamps[i],
                                "Dauer (Sekunden)": duration
                            })
                            current_status = status_history[i]
                            start_idx = i
                    
                    # Record the last status 记录最后一个状态
                    duration = video_timestamps[-1] - video_timestamps[start_idx]
                    status_periods.append({
                        "Maschinenstatus": current_status,
                        "Startzeit": formatted_video_timestamps[start_idx],
                        "Endzeit": formatted_video_timestamps[-1],
                        "Dauer (Sekunden)": duration
                    })
                
                status_periods_df = pd.DataFrame(status_periods)
                st.session_state.status_periods_df = status_periods_df
                
                # Calculate duration for each category (using actual video time) 计算各类别持续时间（使用实际视频时间）
                state_categories = {
                    "Tatsächliche Produktionszeit": [STATUS["WORKING"]],
                    "Ungeplante Stillstände": [
                        STATUS["WAITING_DOOR_CLOSED"],
                        STATUS["WAITING_DOOR_OPEN"],
                        STATUS["HAKEN_ENTFERNEN"],
                        STATUS["WERKZEUG_JUSTIEREN"],
                        STATUS["DRUCKLUFT_ENTFERNEN"],
                        STATUS["HAND_ENTFERNEN"]
                    ],
                    "Geplante Stillstände": [STATUS["VERFAHRBEWEGUNG"]]
                }
                
                # Calculate the duration of each category (based on all analyzed frames) 计算每个类别的持续时间（基于分析的所有帧）
                category_duration = {}
                for category, states in state_categories.items():
                    durations = []
                    for period in status_periods:
                        if period["Maschinenstatus"] in states:
                            durations.append(period["Dauer (Sekunden)"])
                    category_duration[category] = sum(durations)
                
                st.session_state.category_duration = category_duration
    
    # Display historical data and analysis results 显示历史数据和分析结果
    with tab2:
        st.header("Historische Daten")
        
        if st.session_state.has_analyzed:
            if video_path != 0 or st.session_state.video_duration > 0:
                video_duration = st.session_state.video_duration
                minutes = int(video_duration // 60)
                seconds = int(video_duration % 60)
                st.info(f"Videodauer: {minutes:02d}:{seconds:02d}")
                
            
            st.subheader("Detaillierte Aufzeichnungen")
            if st.session_state.history_df is not None:
                # Display data, but hide the original seconds column 显示数据，但隐藏原始秒数列
                display_df = st.session_state.history_df.drop(columns=["Videozeit (Sekunden)"], errors='ignore')
                st.dataframe(display_df, width=True)
            
            
            st.subheader("Statusdauer-Statistik")
            if st.session_state.status_periods_df is not None:
                st.dataframe(st.session_state.status_periods_df, width=True)
            
            
            # Draw status time series chart 绘制状态时间序列图
            st.subheader("Maschinenstatus-Zeitserie")
            
            if len(st.session_state.status_history) > 0 and len(st.session_state.video_timestamps) > 0:
                # Create mapping of status to numeric value 创建状态转换为数值的映射
                unique_statuses = list(set(st.session_state.status_history))
                
                # Create color mapping 创建颜色映射
                colors = px.colors.qualitative.Plotly
                color_mapping = {status: colors[i % len(colors)] for i, status in enumerate(unique_statuses)}
                
                # Create status chart 创建状态图表
                fig = go.Figure()
                
                # Add lines for each status 添加每个状态的线
                for status in unique_statuses:
                    mask = [s == status for s in st.session_state.status_history]
                    if any(mask):
                        # Use video time as X-axis 使用视频时间作为X轴
                        x_values = [st.session_state.video_timestamps[i] for i in range(len(st.session_state.video_timestamps)) if mask[i]]
                        
                        fig.add_trace(go.Scatter(
                            x=x_values,
                            y=[status] * len(x_values),
                            mode='markers',
                            name=status,
                            marker=dict(color=color_mapping[status], size=10)
                        ))
                
                fig.update_layout(
                    title="Maschinenstatus im Zeitverlauf",
                    xaxis_title="Videozeit (Sekunden)",
                    yaxis_title="Maschinenstatus",
                    yaxis=dict(
                        categoryorder='array',
                        categoryarray=unique_statuses
                    ),
                    height=500
                )
                
                st.plotly_chart(fig, width=True)
            else:
                st.warning("Keine Statusdaten verfügbar für die Visualisierung.")
        else:
            st.info("Bitte führen Sie zuerst eine Analyse durch.")
            
    # Display in OEE analysis tab 在OEE分析选项卡中显示
    with tab3:
        st.header("Verfügbarkeitsanalyse")
        
        if st.session_state.has_analyzed:
            if video_path != 0 or st.session_state.video_duration > 0:
                video_duration = st.session_state.video_duration
                minutes = int(video_duration // 60)
                seconds = int(video_duration % 60)
                st.info(f"Videodauer: {minutes:02d}:{seconds:02d}")
                
                # Add analysis method explanation 添加分析方法说明
                total_status_time = sum(st.session_state.category_duration.values())
                st.info(f"Die Analyse basiert auf {total_status_time:.2f} Sekunden Video. " 
                       f"Bei der Verarbeitung wurde jeder {config['process_every_n_frames']}. Frame analysiert.")
            
            # Create pie chart 创建饼图
            if st.session_state.category_duration:
                fig = px.pie(
                    values=list(st.session_state.category_duration.values()),
                    names=list(st.session_state.category_duration.keys()),
                    title="Maschinenstatusverteilung",
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                st.plotly_chart(fig, width=True)
                
                # Display various time metrics 显示各种时间指标
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Verfügbarkeit", f"{st.session_state.verfugbarkeit:.2f}%")
                col2.metric("Tatsächliche Produktionszeit", f"{st.session_state.category_duration.get('Tatsächliche Produktionszeit', 0):.2f} Sek.")
                col3.metric("Ungeplante Stillstände", f"{st.session_state.category_duration.get('Ungeplante Stillstände', 0):.2f} Sek.")
                col4.metric("Geplante Stillstände", f"{st.session_state.category_duration.get('Geplante Stillstände', 0):.2f} Sek.")
            else:
                st.warning("Keine Kategoriedaten verfügbar für die Analyse.")
        else:
            st.info("Bitte führen Sie zuerst eine Analyse durch.")

if __name__ == "__main__":
    main()
