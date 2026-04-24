import plotly.express as px
import streamlit as st

from src.app_state import load_runtime_data
from src.smart_services import nutrition_structure, student_budget_report


st.title("3. 学生端 - 预算与健康建议")
df = load_runtime_data()
student_ids = sorted(df["student_id"].astype(str).unique().tolist())
student_id = st.selectbox("学生编号", student_ids, index=0)
month_budget = st.number_input("本月预算(元)", min_value=100.0, max_value=3000.0, value=600.0, step=50.0)

report = student_budget_report(df, student_id, month_budget)
c1, c2, c3, c4 = st.columns(4)
c1.metric("已消费", f"{report['spent']:.1f} 元")
c2.metric("剩余预算", f"{report['remaining']:.1f} 元")
c3.metric("建议日均", f"{report['suggest_daily']:.1f} 元/天")
c4.metric("月底预测", f"{report['forecast_month_end']:.1f} 元")
if report["overspend"] > 0:
    st.error(f"预算预警：预计超预算 {report['overspend']:.1f} 元。")
else:
    st.success("预算状态正常。")

if "daily_df" in report:
    fig = px.line(report["daily_df"], x="order_time", y="day_amount", markers=True, title="本月每日消费趋势")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("营养结构分析")
days = st.slider("统计最近天数", 7, 30, 7)
ns = nutrition_structure(df, student_id, recent_days=days)
if ns.empty:
    st.info("近期消费记录不足。")
else:
    fig2 = px.pie(ns, names="healthy_tag", values="quantity", title="饮食结构占比")
    st.plotly_chart(fig2, use_container_width=True)
    top_tag = ns.sort_values("ratio", ascending=False).iloc[0]["healthy_tag"]
    if top_tag in {"油炸", "高糖", "重口"}:
        st.warning("健康建议：近期重口/高油高糖偏多，建议增加清淡和蔬菜类。")
    else:
        st.success("健康建议：当前饮食结构较均衡。")
