from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_preprocess import clean_orders, generate_sample_orders, save_dataframe


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "canteen_orders.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "cleaned_orders.csv"
PREFERENCES_PATH = ROOT / "data" / "processed" / "student_preferences.csv"
FEEDBACK_PATH = ROOT / "data" / "processed" / "dish_feedback.csv"
VOTES_PATH = ROOT / "data" / "processed" / "dish_votes.csv"
ANNOUNCEMENTS_PATH = ROOT / "data" / "processed" / "announcements.csv"


def _ensure_dirs() -> None:
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "sample").mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "figures").mkdir(parents=True, exist_ok=True)

    if not PREFERENCES_PATH.exists():
        pd.DataFrame(columns=["student_id", "dish_name", "preference_type", "create_time"]).to_csv(
            PREFERENCES_PATH, index=False, encoding="utf-8-sig"
        )
    if not FEEDBACK_PATH.exists():
        pd.DataFrame(
            columns=[
                "feedback_id",
                "student_id",
                "dish_name",
                "canteen",
                "taste_score",
                "portion_score",
                "price_score",
                "service_score",
                "repurchase",
                "comment",
                "create_time",
            ]
        ).to_csv(FEEDBACK_PATH, index=False, encoding="utf-8-sig")
    if not VOTES_PATH.exists():
        pd.DataFrame(columns=["vote_id", "student_id", "dish_candidate", "canteen", "reason", "vote_time"]).to_csv(
            VOTES_PATH, index=False, encoding="utf-8-sig"
        )
    if not ANNOUNCEMENTS_PATH.exists():
        pd.DataFrame(columns=["announce_id", "title", "content", "type", "canteen", "create_time"]).to_csv(
            ANNOUNCEMENTS_PATH, index=False, encoding="utf-8-sig"
        )


def load_runtime_data() -> pd.DataFrame:
    _ensure_dirs()
    if "clean_df" in st.session_state:
        return st.session_state["clean_df"]

    if PROCESSED_PATH.exists():
        df = pd.read_csv(PROCESSED_PATH)
        st.session_state["clean_df"] = df
        return df

    sample_raw = generate_sample_orders()
    sample_clean = clean_orders(sample_raw)
    save_dataframe(sample_raw, RAW_PATH)
    save_dataframe(sample_clean, PROCESSED_PATH)
    save_dataframe(sample_raw, ROOT / "data" / "sample" / "sample_orders.csv")
    st.session_state["clean_df"] = sample_clean
    return sample_clean


def update_runtime_data(clean_df: pd.DataFrame, raw_df: pd.DataFrame | None = None) -> None:
    _ensure_dirs()
    st.session_state["clean_df"] = clean_df
    save_dataframe(clean_df, PROCESSED_PATH)
    if raw_df is not None:
        save_dataframe(raw_df, RAW_PATH)


def reset_to_sample_data() -> None:
    sample_raw = generate_sample_orders()
    sample_clean = clean_orders(sample_raw)
    update_runtime_data(sample_clean, sample_raw)
