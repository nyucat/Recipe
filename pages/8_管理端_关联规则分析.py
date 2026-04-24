import plotly.express as px
import streamlit as st

from src.app_state import load_runtime_data
from src.association_rules import mine_association_rules, recommend_by_selected_dish


st.title("8. 管理端 - 关联规则分析")
df = load_runtime_data()

c1, c2, c3 = st.columns(3)
min_support = c1.slider("最小支持度", 0.001, 0.10, 0.008, 0.001)
min_conf = c2.slider("最小置信度", 0.05, 0.90, 0.20, 0.05)
min_lift = c3.slider("最小提升度", 1.0, 3.0, 1.1, 0.1)

rules = mine_association_rules(df, min_support=min_support, min_confidence=min_conf, min_lift=min_lift)
if rules.empty:
    st.warning("当前阈值下未找到有效规则。")
else:
    st.success(f"共挖掘 {len(rules)} 条规则。")
    st.dataframe(rules.head(30), use_container_width=True)
    fig = px.scatter(rules.head(200), x="support", y="confidence", size="lift", hover_data=["antecedents_str", "consequents_str"], title="规则分布")
    st.plotly_chart(fig, use_container_width=True)

    dish_name = st.selectbox("选择菜品查看搭配", sorted(df["dish_name"].dropna().unique().tolist()))
    rec = recommend_by_selected_dish(rules, dish_name, top_n=8)
    if rec.empty:
        st.info("该菜品暂无明显搭配规则。")
    else:
        st.dataframe(rec[["antecedents_str", "consequents_str", "confidence", "lift", "reason"]], use_container_width=True)
