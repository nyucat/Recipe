import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_state import FEEDBACK_PATH, load_runtime_data
from src.sales_predict import build_meal_plan, forecast_next_days, train_sales_model
from src.smart_services import load_table


st.title("11. 后勤端 - 多食堂对比")
df = load_runtime_data()
df["order_time"] = pd.to_datetime(df["order_time"], errors="coerce")

cmp = (
    df.groupby("canteen", as_index=False)
    .agg(sales=("amount", "sum"), orders=("order_id", "nunique"), students=("student_id", "nunique"), avg_rating=("rating", "mean"))
    .sort_values("sales", ascending=False)
)
st.dataframe(cmp, use_container_width=True)
fig = px.bar(cmp, x="canteen", y=["sales", "orders"], barmode="group", title="多食堂经营规模对比")
st.plotly_chart(fig, use_container_width=True)

fb = load_table(
    FEEDBACK_PATH,
    ["feedback_id", "student_id", "dish_name", "canteen", "taste_score", "portion_score", "price_score", "service_score", "repurchase", "comment", "create_time"],
)
if not fb.empty:
    fb["satisfaction"] = fb[["taste_score", "portion_score", "price_score", "service_score"]].mean(axis=1)
    sat = fb.groupby("canteen", as_index=False).agg(satisfaction=("satisfaction", "mean"), feedbacks=("feedback_id", "count"))
    st.dataframe(sat.sort_values("satisfaction"), use_container_width=True)

pack = train_sales_model(df)
future7 = forecast_next_days(pack, days=7)
plan = build_meal_plan(future7)
risk = plan["risk"].value_counts().rename_axis("risk").reset_index(name="count")
fig2 = px.pie(risk, names="risk", values="count", title="全局缺货/浪费风险结构")
st.plotly_chart(fig2, use_container_width=True)
