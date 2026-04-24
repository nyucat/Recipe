import plotly.express as px
import streamlit as st

from src.app_state import PREFERENCES_PATH, load_runtime_data
from src.smart_services import build_today_recommendation, crowding_prediction, load_table


st.title("1. 学生端 - 今日吃什么")
df = load_runtime_data()
student_ids = sorted(df["student_id"].astype(str).unique().tolist())

c1, c2, c3 = st.columns(3)
student_id = c1.selectbox("学生编号", student_ids, index=0)
budget = c2.selectbox("预算", ["10元以内", "10-15元", "15-20元", "不限"], index=1)
period = c3.selectbox("时段", ["早餐", "午餐", "晚餐", "夜宵"], index=1)

c4, c5, c6 = st.columns(3)
taste = c4.selectbox("口味", ["不限", "清淡", "辣", "甜", "咸"], index=0)
goal = c5.selectbox("目标", ["吃饱", "减脂", "高蛋白", "省钱", "尝鲜"], index=0)
canteen_pref = c6.selectbox("食堂", ["不限"] + sorted(df["canteen"].unique().tolist()), index=0)

pref = load_table(PREFERENCES_PATH, ["student_id", "dish_name", "preference_type", "create_time"])
disliked = set(
    pref[(pref["student_id"].astype(str) == student_id) & (pref["preference_type"] == "dislike")]["dish_name"]
    .astype(str)
    .tolist()
)
rec = build_today_recommendation(df, student_id, budget, taste, goal, period, canteen_pref, disliked_dishes=disliked, top_n=8)
if rec.empty:
    st.warning("当前筛选条件下暂无推荐结果。")
else:
    show = rec[["dish_name", "avg_price", "canteen", "window", "avg_rating", "reason"]].copy()
    show.columns = ["推荐菜品", "价格", "食堂", "窗口", "评分", "推荐理由"]
    st.dataframe(show, use_container_width=True)

st.subheader("拥挤度与避峰建议")
crowd = crowding_prediction(df)
if crowd.empty:
    st.info("暂无足够时段数据。")
else:
    st.dataframe(crowd[["canteen", "pred_order", "level", "queue_time"]], use_container_width=True)
    fig = px.bar(crowd, x="canteen", y="pred_order", color="level", text="queue_time", title="食堂拥挤度预测")
    st.plotly_chart(fig, use_container_width=True)
