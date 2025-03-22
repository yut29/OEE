import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# Configure page
st.set_page_config(page_title="Stillstandserkennung", layout="wide")

st.title("Zerspanungsmaschine-Stillstandserkennung")

st.write("这是一个简化版应用，用于测试 Streamlit Cloud 环境。")

# Check environment
st.write(f"Python 版本: {os.sys.version}")
st.write(f"OpenCV 版本: {cv2.__version__}")
st.write(f"NumPy 版本: {np.__version__}")
st.write(f"Pandas 版本: {pd.__version__}")

# Try to import ultralytics
try:
    from ultralytics import YOLO
    st.success("成功导入 ultralytics YOLO!")
    
    # Try to initialize a model
    try:
        # For testing, we can use a small pre-trained model
        model = YOLO("yolov8n.pt")  # 使用小型预训练模型进行测试
        st.success("成功初始化 YOLO 模型!")
    except Exception as e:
        st.error(f"初始化 YOLO 模型时出错: {e}")
except Exception as e:
    st.error(f"导入 ultralytics 时出错: {e}")

# Display system info
st.subheader("系统信息")
st.write(f"当前工作目录: {os.getcwd()}")
st.write(f"环境变量: {dict(os.environ)}")
