import plotly.express as px
import streamlit as st

from src.app_state import load_runtime_data
from src.sales_predict import forecast_next_days, train_sales_model


st.title("9. 管理端 - 销量预测")
df = load_runtime_data()
model_pack = train_sales_model(df)
metrics = model_pack["metrics"]

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{metrics['MAE']:.3f}")
c2.metric("RMSE", f"{metrics['RMSE']:.3f}")
c3.metric("R²", f"{metrics['R2']:.3f}")

eval_df = model_pack["eval_df"].copy()
by_day = eval_df.groupby("date", as_index=False).agg(real=("quantity", "sum"), pred=("pred", "sum"))
fig_eval = px.line(by_day, x="date", y=["real", "pred"], markers=True, title="真实值 vs 预测值")
st.plotly_chart(fig_eval, use_container_width=True)

days = st.slider("预测未来天数", 3, 14, 7)
future = forecast_next_days(model_pack, days=days)
st.dataframe(future.head(30), use_container_width=True)

dish = st.selectbox("查看单菜品预测", sorted(future["dish_name"].unique().tolist()))
dish_future = future[future["dish_name"] == dish]
fig_future = px.line(dish_future, x="date", y="pred_quantity", markers=True, title=f"{dish} 未来 {days} 天销量趋势")
st.plotly_chart(fig_future, use_container_width=True)
