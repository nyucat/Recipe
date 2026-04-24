from pathlib import Path

import streamlit as st

from src.app_state import load_runtime_data, reset_to_sample_data


st.set_page_config(
    page_title="校园食堂商务智能分析系统",
    page_icon="🍽️",
    layout="wide",
)

st.title("校园食堂消费分析与菜品推荐系统")
st.caption("《商务智能》课程大作业演示系统（描述性 + 预测性 + 规范性分析）")

df = load_runtime_data()

col1, col2, col3 = st.columns(3)
col1.metric("当前数据记录数", f"{len(df):,}")
col2.metric("匿名学生数", f"{df['student_id'].nunique():,}")
col3.metric("菜品数", f"{df['dish_name'].nunique():,}")

st.markdown(
    """
### 使用说明
1. 左侧菜单已按三类用户重构为 12 个页面（学生端、管理端、后勤端）。  
2. 建议演示顺序：`1-4 学生端 -> 5-10 管理端 -> 11-12 后勤端`。  
3. 所有页面共享同一份运行数据并联动分析结果。  
"""
)

with st.expander("项目文件位置"):
    root = Path(__file__).resolve().parent
    st.code(str(root), language="text")

if st.button("重置为系统样例数据", type="secondary"):
    reset_to_sample_data()
    st.success("已重置为样例数据，请前往各页面查看。")
