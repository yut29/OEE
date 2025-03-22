import streamlit as st
import os
import platform
import sys

st.set_page_config(page_title="环境测试", layout="wide")

st.title("Streamlit 环境测试")

st.write(f"Python 版本: {platform.python_version()}")
st.write(f"Python 路径: {sys.executable}")
st.write(f"当前工作目录: {os.getcwd()}")

st.subheader("导入测试")

# 基本库测试
with st.expander("基本库"):
    try:
        import numpy as np
        st.success(f"NumPy 导入成功，版本: {np.__version__}")
    except Exception as e:
        st.error(f"NumPy 导入失败: {e}")
    
    try:
        import pandas as pd
        st.success(f"Pandas 导入成功，版本: {pd.__version__}")
    except Exception as e:
        st.error(f"Pandas 导入失败: {e}")
    
    try:
        import plotly
        st.success(f"Plotly 导入成功，版本: {plotly.__version__}")
    except Exception as e:
        st.error(f"Plotly 导入失败: {e}")

# OpenCV 测试
with st.expander("OpenCV"):
    try:
        import cv2
        st.success(f"OpenCV 导入成功，版本: {cv2.__version__}")
    except Exception as e:
        st.error(f"OpenCV 导入失败: {e}")

st.subheader("文件列表")
st.write("仓库文件：")
for root, dirs, files in os.walk("."):
    if ".git" not in root:  # 排除 .git 目录
        for file in files:
            st.code(os.path.join(root, file))
