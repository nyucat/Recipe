import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_state import load_runtime_data
from src.descriptive_analysis import add_time_features


st.title("6. 管理端 - 菜品销售分析")
df = add_time_features(load_runtime_data())
df["date"] = pd.to_datetime(df["date"])

canteens = ["全部"] + sorted(df["canteen"].dropna().unique().tolist())
pick_canteen = st.selectbox("食堂筛选", canteens, index=0)
date_min = df["date"].min().date()
date_max = df["date"].max().date()
start_date, end_date = st.date_input("日期范围", value=(date_min, date_max), min_value=date_min, max_value=date_max)

flt = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]
if pick_canteen != "全部":
    flt = flt[flt["canteen"] == pick_canteen]

top = (
    flt.groupby("dish_name", as_index=False)
    .agg(sales_qty=("quantity", "sum"), sales_amount=("amount", "sum"), avg_rating=("rating", "mean"))
    .sort_values("sales_qty", ascending=False)
)
fig_top = px.bar(top.head(12), x="dish_name", y="sales_qty", title="热门菜品 TOP12", text_auto=True)
st.plotly_chart(fig_top, use_container_width=True)

price_qty = flt.groupby("dish_name", as_index=False).agg(avg_price=("price", "mean"), sales_qty=("quantity", "sum"), avg_rating=("rating", "mean"))
fig_scatter = px.scatter(price_qty, x="avg_price", y="sales_qty", size="avg_rating", hover_name="dish_name", title="价格-销量关系")
st.plotly_chart(fig_scatter, use_container_width=True)

c1, c2 = st.columns(2)
c1.dataframe(top.head(10), use_container_width=True)
c2.dataframe(top.tail(10).sort_values("sales_qty"), use_container_width=True)
