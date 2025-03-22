import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time

st.set_page_config(page_title="视频上传测试", layout="wide")

st.title("视频上传与处理测试")

# 显示当前环境信息
st.subheader("环境信息")
import sys
import platform

col1, col2 = st.columns(2)
with col1:
    st.info(f"Python 版本: {platform.python_version()}")
    st.info(f"OpenCV 版本: {cv2.__version__}")
with col2:
    st.info(f"NumPy 版本: {np.__version__}")
    try:
        import torch
        st.info(f"PyTorch 版本: {torch.__version__}")
    except:
        st.warning("PyTorch 未加载")

# 视频上传部分
st.subheader("上传视频文件")
uploaded_file = st.file_uploader("选择一个视频文件", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # 显示上传的文件信息
    file_details = {
        "文件名": uploaded_file.name,
        "文件类型": uploaded_file.type,
        "文件大小": f"{len(uploaded_file.getbuffer())/1024/1024:.2f} MB"
    }
    st.write("文件详情:")
    for key, value in file_details.items():
        st.write(f"- {key}: {value}")
    
    # 保存上传的视频到临时文件
    st.write("正在保存视频文件...")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
    temp_file.write(uploaded_file.getbuffer())
    video_path = temp_file.name
    temp_file.close()
    
    st.write(f"临时文件路径: {video_path}")
    st.write(f"文件是否存在: {os.path.exists(video_path)}")
    
    # 尝试使用Streamlit的原生视频播放器
    st.subheader("使用Streamlit视频播放器")
    try:
        st.video(video_path)
        st.success("Streamlit原生视频播放器成功加载视频")
    except Exception as e:
        st.error(f"Streamlit视频播放失败: {e}")
    
    # 第一部分：尝试用OpenCV打开视频并获取基本信息
    st.subheader("OpenCV视频信息")
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("OpenCV无法打开视频文件")
        else:
            # 获取视频基本信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            
            # 将fourcc转换为可读字符
            fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            # 显示视频信息
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("视频宽度", f"{width}px")
                st.metric("视频高度", f"{height}px")
                st.metric("帧率(FPS)", f"{fps:.2f}")
            with info_col2:
                st.metric("总帧数", frame_count)
                st.metric("视频长度", f"{frame_count/fps:.2f}秒")
                st.metric("编解码器", fourcc_str)
            
            st.success("OpenCV成功读取视频基本信息")
            
            # 第二部分：尝试读取视频帧
            st.subheader("OpenCV帧读取测试")
            
            # 读取第一帧
            ret, frame = cap.read()
            
            if not ret:
                st.error("无法读取视频帧")
            else:
                # 显示读取的帧的详细信息
                st.write(f"帧形状: {frame.shape}")
                st.write(f"帧类型: {frame.dtype}")
                st.write(f"帧是否为空: {frame.size == 0}")
                
                if frame.size > 0 and frame.shape[0] > 0 and frame.shape[1] > 0:
                    # 转换BGR到RGB以供显示
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="第一帧", use_column_width=True)
                    st.success("成功读取并显示第一帧")
                else:
                    st.error(f"帧尺寸无效: {frame.shape}")
            
            # 第三部分：读取多帧并创建缩略图
            st.subheader("多帧读取测试")
            
            # 重置视频
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # 读取最多5帧，间隔为总帧数的1/5
            if frame_count > 5:
                step = max(1, int(frame_count / 5))
            else:
                step = 1
            
            frames = []
            positions = []
            progress_bar = st.progress(0)
            
            for i in range(min(5, int(frame_count))):
                # 设置帧位置
                frame_pos = i * step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                
                # 读取帧
                ret, frame = cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    # 转换BGR到RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    positions.append(frame_pos)
                
                # 更新进度条
                progress_bar.progress((i+1)/min(5, int(frame_count)))
            
            # 显示帧缩略图
            if frames:
                st.write(f"成功读取 {len(frames)} 帧")
                cols = st.columns(len(frames))
                for i, (frame, pos) in enumerate(zip(frames, positions)):
                    with cols[i]:
                        st.image(frame, caption=f"帧 #{pos}", use_column_width=True)
                st.success("成功读取并显示多个帧")
            else:
                st.error("无法读取任何帧")
            
            # 关闭视频
            cap.release()
    except Exception as e:
        st.error(f"OpenCV处理时出错: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    # 清理临时文件
    try:
        os.unlink(video_path)
        st.write("已清理临时文件")
    except:
        st.write("无法清理临时文件")

# 添加测试视频生成功能
st.subheader("生成测试视频")
if st.button("创建并测试样本视频"):
    with st.spinner("正在创建测试视频..."):
        # 创建一个简单的测试视频
        test_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # 设置视频参数
        width, height = 640, 480
        fps = 30
        seconds = 3
        
        # 初始化VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编解码器
        out = cv2.VideoWriter(test_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            st.error("无法创建视频文件")
        else:
            # 生成3秒的视频
            for i in range(fps * seconds):
                # 创建带有移动方块的帧
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                # 在背景上加上时间文本
                cv2.putText(frame, f"Frame: {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # 在不同位置绘制移动的方块
                x = int((i % fps) * (width - 100) / fps)
                cv2.rectangle(frame, (x, height//2-50), (x+100, height//2+50), (0, 255, 0), -1)
                # 写入帧
                out.write(frame)
            
            # 释放资源
            out.release()
            
            st.success(f"生成的测试视频路径: {test_video_path}")
            
            # 显示测试视频
            st.video(test_video_path)
            
            # 使用OpenCV打开测试视频
            try:
                cap = cv2.VideoCapture(test_video_path)
                if cap.isOpened():
                    st.success("OpenCV成功打开测试视频")
                    
                    # 读取并显示第一帧
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="测试视频第一帧", use_column_width=True)
                    else:
                        st.error("无法读取测试视频帧")
                    
                    cap.release()
                else:
                    st.error("OpenCV无法打开测试视频")
            except Exception as e:
                st.error(f"处理测试视频时出错: {e}")
