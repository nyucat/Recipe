import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_state import load_runtime_data
from src.descriptive_analysis import add_time_features


st.title("7. 管理端 - 消费高峰分析")
df = add_time_features(load_runtime_data())
order_level = df.groupby("order_id", as_index=False).agg(order_time=("order_time", "first"), amount=("amount", "sum"), period=("period", "first"))
order_level["order_time"] = pd.to_datetime(order_level["order_time"])
order_level["half_hour"] = order_level["order_time"].dt.floor("30min").dt.strftime("%H:%M")
order_level["weekday_cn"] = order_level["order_time"].dt.weekday.map({0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"})

period_stat = order_level.groupby("period", as_index=False).agg(orders=("order_id", "nunique"), sales=("amount", "sum"))
fig = px.bar(period_stat, x="period", y=["orders", "sales"], barmode="group", title="时段订单与销售")
st.plotly_chart(fig, use_container_width=True)

hh = order_level.groupby("half_hour", as_index=False).agg(orders=("order_id", "nunique")).sort_values("half_hour")
fig2 = px.line(hh, x="half_hour", y="orders", markers=True, title="30分钟粒度订单趋势")
st.plotly_chart(fig2, use_container_width=True)

heat = order_level.groupby(["weekday_cn", order_level["order_time"].dt.hour], as_index=False).agg(orders=("order_id", "nunique"))
heat.columns = ["weekday_cn", "hour", "orders"]
fig3 = px.density_heatmap(heat, x="hour", y="weekday_cn", z="orders", color_continuous_scale="YlOrRd", title="星期-小时热力图")
st.plotly_chart(fig3, use_container_width=True)

peak = hh.sort_values("orders", ascending=False).iloc[0]
st.success(f"高峰提示：订单高峰约在 `{peak['half_hour']}`，建议高峰前完成主菜备餐。")
