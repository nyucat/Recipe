import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_state import load_runtime_data
from src.descriptive_analysis import add_time_features


st.title("2. 学生端 - 我的消费报告")
df = add_time_features(load_runtime_data())
student_ids = sorted(df["student_id"].astype(str).unique().tolist())
student_id = st.selectbox("选择学生编号", student_ids, index=0)

stu = df[df["student_id"].astype(str) == student_id].copy()
if stu.empty:
    st.info("该学生暂无数据。")
    st.stop()

order_level = stu.groupby("order_id", as_index=False).agg(order_amount=("amount", "sum"), order_time=("order_time", "first"))
c1, c2, c3, c4 = st.columns(4)
c1.metric("累计消费", f"{stu['amount'].sum():.1f} 元")
c2.metric("消费次数", f"{stu['order_id'].nunique():.0f} 单")
c3.metric("平均客单价", f"{order_level['order_amount'].mean():.2f} 元")
c4.metric("常去食堂", stu["canteen"].value_counts().index[0])

daily = stu.groupby(stu["order_time"].dt.date, as_index=False).agg(day_amount=("amount", "sum"))
fig_daily = px.line(daily, x="order_time", y="day_amount", markers=True, title="每日消费趋势")
st.plotly_chart(fig_daily, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    top_dish = stu.groupby("dish_name", as_index=False).agg(freq=("order_id", "nunique"), amount=("amount", "sum")).sort_values("freq", ascending=False)
    st.subheader("常吃菜品")
    st.dataframe(top_dish.head(10), use_container_width=True)
with c2:
    period = stu.groupby("period", as_index=False).agg(orders=("order_id", "nunique")).sort_values("orders", ascending=False)
    fig_period = px.pie(period, names="period", values="orders", title="消费时段占比")
    st.plotly_chart(fig_period, use_container_width=True)

tags = []
peak = period.iloc[0]["period"] if not period.empty else ""
if peak == "早餐":
    tags.append("早餐党")
if peak == "夜宵":
    tags.append("夜宵党")
if (stu["price"] <= 10).mean() > 0.5:
    tags.append("省钱达人")
if stu["category"].astype(str).str.contains("主食").mean() > 0.5:
    tags.append("主食偏好")
if not tags:
    tags.append("均衡消费型")
st.success("用户标签：" + "、".join(tags))
