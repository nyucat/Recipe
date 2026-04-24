from pathlib import Path

import pandas as pd
import streamlit as st

from src.app_state import ROOT, load_runtime_data
from src.report_generator import generate_markdown_report
from src.sales_predict import build_meal_plan, forecast_next_days, train_sales_model


st.title("12. 后勤端 - 自动报告生成")
df = load_runtime_data()
df["order_time"] = pd.to_datetime(df["order_time"], errors="coerce")
df["date"] = df["order_time"].dt.date

report_type = st.selectbox("报告类型", ["每日报告", "每周报告", "每月报告"])
if report_type == "每日报告":
    part = df[df["date"] == df["date"].max()].copy()
elif report_type == "每周报告":
    part = df[df["order_time"] >= (df["order_time"].max() - pd.Timedelta(days=7))].copy()
else:
    part = df[df["order_time"] >= (df["order_time"].max() - pd.Timedelta(days=30))].copy()

kpi = {
    "total_sales": float(part["amount"].sum()),
    "total_orders": float(part["order_id"].nunique()),
    "total_students": float(part["student_id"].nunique()),
    "avg_ticket": float(part.groupby("order_id")["amount"].sum().mean() if not part.empty else 0),
}
top_dishes = (
    part.groupby("dish_name", as_index=False)
    .agg(sales_qty=("quantity", "sum"), avg_rating=("rating", "mean"))
    .sort_values("sales_qty", ascending=False)
)
peak_period = (
    part.assign(hour=part["order_time"].dt.hour, period=lambda x: x["hour"].map(lambda h: "早餐" if 6 <= h < 10 else "午餐" if 10 <= h < 14 else "晚餐" if 16 <= h < 20 else "夜宵"))
    .groupby("period")["order_id"]
    .nunique()
    .sort_values(ascending=False)
    .index[0]
    if not part.empty
    else "无"
)

pack = train_sales_model(df)
future7 = forecast_next_days(pack, days=7)
plan = build_meal_plan(future7)
predict_brief = f"未来 7 天预测销量 {future7['pred_quantity'].sum():.0f} 份，建议备餐 {plan['suggested_prep'].sum():.0f} 份。"
report = generate_markdown_report(kpi, top_dishes, peak_period, "套餐策略详见关联规则分析页。", "群体策略详见学生画像与学生端报告。", predict_brief)

st.markdown(report)
path = ROOT / "outputs" / "reports" / "auto_report.md"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(report, encoding="utf-8")
st.caption(f"报告已保存到：{path}")
