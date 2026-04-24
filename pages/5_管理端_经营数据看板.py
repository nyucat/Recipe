import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_state import load_runtime_data
from src.smart_services import suggest_promotions


st.title("5. 管理端 - 经营数据看板")
df = load_runtime_data()
df["order_time"] = pd.to_datetime(df["order_time"], errors="coerce")
today = df["order_time"].dt.date.max()
today_df = df[df["order_time"].dt.date == today]

c1, c2, c3, c4 = st.columns(4)
c1.metric("今日销售额", f"{today_df['amount'].sum():.1f} 元")
c2.metric("今日订单数", f"{today_df['order_id'].nunique():.0f} 单")
c3.metric("今日消费人数", f"{today_df['student_id'].nunique():.0f} 人")
c4.metric("今日客单价", f"{today_df.groupby('order_id')['amount'].sum().mean():.2f} 元" if not today_df.empty else "0")

daily = df.groupby(df["order_time"].dt.date, as_index=False).agg(sales=("amount", "sum"), orders=("order_id", "nunique"))
fig = px.line(daily, x="order_time", y=["sales", "orders"], markers=True, title="经营趋势")
st.plotly_chart(fig, use_container_width=True)

top_window = today_df.groupby("window", as_index=False).agg(orders=("order_id", "nunique")).sort_values("orders", ascending=False)
fig2 = px.bar(top_window.head(10), x="window", y="orders", title="今日窗口订单排行", text_auto=True)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("促销建议")
promo = suggest_promotions(df)
st.dataframe(promo[["dish_name", "recent_qty", "prev_qty", "decline_ratio", "promo_type", "advice"]].head(15), use_container_width=True)
