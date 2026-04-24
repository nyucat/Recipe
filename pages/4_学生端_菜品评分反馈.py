from datetime import datetime

import pandas as pd
import streamlit as st

from src.app_state import FEEDBACK_PATH, PREFERENCES_PATH, VOTES_PATH, load_runtime_data
from src.smart_services import load_table, save_table


st.title("4. 学生端 - 菜品评分反馈")
df = load_runtime_data()
student_ids = sorted(df["student_id"].astype(str).unique().tolist())
student_id = st.selectbox("学生编号", student_ids, index=0)

tab1, tab2, tab3 = st.tabs(["偏好设置", "评分反馈", "上新投票"])

with tab1:
    dish = st.selectbox("菜品", sorted(df["dish_name"].unique().tolist()))
    pref_type = st.radio("偏好类型", ["like", "dislike"], horizontal=True)
    if st.button("保存偏好"):
        pref = load_table(PREFERENCES_PATH, ["student_id", "dish_name", "preference_type", "create_time"])
        pref = pref[~((pref["student_id"].astype(str) == student_id) & (pref["dish_name"] == dish))]
        pref = pd.concat(
            [pref, pd.DataFrame([{"student_id": student_id, "dish_name": dish, "preference_type": pref_type, "create_time": datetime.now().isoformat()}])],
            ignore_index=True,
        )
        save_table(pref, PREFERENCES_PATH)
        st.success("偏好已保存。")
    pref_view = load_table(PREFERENCES_PATH, ["student_id", "dish_name", "preference_type", "create_time"])
    st.dataframe(pref_view[pref_view["student_id"].astype(str) == student_id], use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    dish_fb = c1.selectbox("评分菜品", sorted(df["dish_name"].unique().tolist()))
    canteen_fb = c2.selectbox("食堂", sorted(df["canteen"].unique().tolist()))
    c3, c4, c5, c6 = st.columns(4)
    taste_score = c3.slider("口味", 1, 5, 4)
    portion_score = c4.slider("分量", 1, 5, 4)
    price_score = c5.slider("价格", 1, 5, 4)
    service_score = c6.slider("服务", 1, 5, 4)
    repurchase = st.radio("再次购买", ["是", "否"], horizontal=True)
    comment = st.text_input("评论")
    if st.button("提交评分反馈"):
        fb = load_table(
            FEEDBACK_PATH,
            ["feedback_id", "student_id", "dish_name", "canteen", "taste_score", "portion_score", "price_score", "service_score", "repurchase", "comment", "create_time"],
        )
        row = {
            "feedback_id": f"FB{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "student_id": student_id,
            "dish_name": dish_fb,
            "canteen": canteen_fb,
            "taste_score": taste_score,
            "portion_score": portion_score,
            "price_score": price_score,
            "service_score": service_score,
            "repurchase": repurchase,
            "comment": comment,
            "create_time": datetime.now().isoformat(),
        }
        fb = pd.concat([fb, pd.DataFrame([row])], ignore_index=True)
        save_table(fb, FEEDBACK_PATH)
        st.success("反馈提交成功。")

with tab3:
    vote_dish = st.selectbox("希望新增菜品", ["重庆小面", "烤盘饭", "轻食沙拉", "麻辣烫", "粤式烧腊", "韩式拌饭"])
    vote_reason = st.text_input("理由")
    if st.button("提交投票"):
        votes = load_table(VOTES_PATH, ["vote_id", "student_id", "dish_candidate", "canteen", "reason", "vote_time"])
        votes = pd.concat(
            [
                votes,
                pd.DataFrame([{"vote_id": f"V{datetime.now().strftime('%Y%m%d%H%M%S')}", "student_id": student_id, "dish_candidate": vote_dish, "canteen": "全部", "reason": vote_reason, "vote_time": datetime.now().isoformat()}]),
            ],
            ignore_index=True,
        )
        save_table(votes, VOTES_PATH)
        st.success("投票成功。")
    votes = load_table(VOTES_PATH, ["vote_id", "student_id", "dish_candidate", "canteen", "reason", "vote_time"])
    if not votes.empty:
        st.dataframe(votes.groupby("dish_candidate", as_index=False).agg(votes=("vote_id", "count")).sort_values("votes", ascending=False), use_container_width=True)
