import plotly.express as px
import streamlit as st

from src.app_state import load_runtime_data
from src.sales_predict import build_meal_plan, forecast_next_days, train_sales_model


st.title("10. 管理端 - 备餐建议")
df = load_runtime_data()
pack = train_sales_model(df)
days = st.slider("备餐规划天数", 1, 14, 7)
future = forecast_next_days(pack, days=days)
plan = build_meal_plan(future)

st.dataframe(plan[["date", "dish_name", "window", "pred_quantity", "suggested_prep", "risk", "advice"]], use_container_width=True)

risk = plan.groupby("risk", as_index=False).agg(cnt=("dish_name", "count"))
fig = px.pie(risk, names="risk", values="cnt", title="风险分布")
st.plotly_chart(fig, use_container_width=True)

window_plan = plan.groupby(["date", "window"], as_index=False).agg(pred_total=("pred_quantity", "sum"), prep_total=("suggested_prep", "sum"))
st.subheader("窗口级备餐计划")
st.dataframe(window_plan, use_container_width=True)
