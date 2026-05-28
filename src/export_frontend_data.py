from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.association_rules import mine_association_rules
from src.alert_center import build_alert_center
from src.comment_analysis import build_comment_keyword_analysis
from src.data_preprocess import clean_orders, generate_sample_orders
from src.descriptive_analysis import add_time_features, compute_kpis
from src.frontend_analytics import (
    build_combo_suggestions,
    build_feedback_insights,
    build_lifecycle,
    build_resource_utilization,
    json_ready_records,
)
from src.logistics_analytics import (
    build_canteen_efficiency,
    build_category_structure,
    build_decision_actions,
    build_report_sections,
    build_risk_overview,
    build_satisfaction_trend,
)
from src.promo_evaluation import build_promo_evaluation
from src.recommender import recommend_for_student, recommend_hot_dishes
from src.sales_predict import build_meal_plan, forecast_next_days, train_sales_model
from src.smart_services import crowding_prediction, nutrition_structure, quality_score_report, student_budget_report, suggest_promotions
from src.strategy_simulator import build_strategy_baseline
from src.user_cluster import train_user_clusters


PROCESSED_PATH = ROOT / "data" / "processed" / "cleaned_orders.csv"
FEEDBACK_PATH = ROOT / "data" / "processed" / "dish_feedback.csv"
PREFERENCES_PATH = ROOT / "data" / "processed" / "student_preferences.csv"
VOTES_PATH = ROOT / "data" / "processed" / "dish_votes.csv"
ANNOUNCEMENTS_PATH = ROOT / "data" / "processed" / "announcements.csv"
FRONTEND_DATA_PATH = ROOT / "frontend" / "public" / "dashboard-data.json"


def _read_or_bootstrap_orders() -> pd.DataFrame:
    if PROCESSED_PATH.exists():
        return pd.read_csv(PROCESSED_PATH)
    sample = clean_orders(generate_sample_orders())
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")
    return sample


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _default_announcements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"title": "一食堂新菜试营业", "content": "番茄牛腩饭本周上线，欢迎体验并反馈口味。", "type": "上新", "canteen": "一食堂"},
            {"title": "二食堂晚餐优惠", "content": "牛肉面晚餐时段立减 2 元，适合避峰快速取餐。", "type": "活动", "canteen": "二食堂"},
            {"title": "三食堂窗口调整", "content": "轻食窗口周五中午暂停营业，请留意公告安排。", "type": "通知", "canteen": "三食堂"},
        ]
    )


def build_dashboard_payload() -> dict:
    orders = _read_or_bootstrap_orders()
    orders["order_time"] = pd.to_datetime(orders["order_time"], errors="coerce")
    data = add_time_features(orders)

    student_ids = sorted(data["student_id"].astype(str).unique().tolist())
    demo_student = student_ids[0] if student_ids else ""

    kpis = compute_kpis(orders)
    quality = quality_score_report(orders)
    promotions = suggest_promotions(orders)
    hot_dishes = recommend_hot_dishes(orders, top_n=10)
    rules = mine_association_rules(orders, min_support=0.008, min_confidence=0.2, min_lift=1.05)
    crowding = crowding_prediction(orders)

    student_features, cluster_profile, _ = train_user_clusters(orders, n_clusters=4)
    peer_recommendation = recommend_for_student(orders, student_features, demo_student, top_n=6)

    model_pack = train_sales_model(orders)
    future = forecast_next_days(model_pack, days=7)
    meal_plan = build_meal_plan(future)

    daily_sales = (
        data.groupby("date", as_index=False)
        .agg(sales=("amount", "sum"), orders=("order_id", "nunique"))
        .sort_values("date")
    )
    canteen_compare = (
        data.groupby("canteen", as_index=False)
        .agg(
            sales=("amount", "sum"),
            orders=("order_id", "nunique"),
            students=("student_id", "nunique"),
            rating=("rating", "mean"),
        )
        .sort_values("sales", ascending=False)
    )
    canteen_daily_sales = (
        data.groupby(["date", "canteen"], as_index=False)
        .agg(sales=("amount", "sum"), orders=("order_id", "nunique"))
        .sort_values(["date", "canteen"])
    )
    dish_sales = (
        data.groupby("dish_name", as_index=False)
        .agg(
            sales_qty=("quantity", "sum"),
            sales_amount=("amount", "sum"),
            avg_rating=("rating", "mean"),
            avg_price=("price", "mean"),
        )
        .sort_values("sales_qty", ascending=False)
    )
    period_stat = data.groupby("period", as_index=False).agg(orders=("order_id", "nunique"), sales=("amount", "sum"))
    heat = (
        data.groupby(["weekday_cn", "hour"], as_index=False)
        .agg(orders=("order_id", "nunique"))
        .sort_values(["weekday_cn", "hour"])
    )
    half_hour = (
        data.assign(half_hour=pd.to_datetime(data["order_time"]).dt.floor("30min").dt.strftime("%H:%M"))
        .groupby("half_hour", as_index=False)
        .agg(orders=("order_id", "nunique"))
        .sort_values("half_hour")
    )
    window_pressure = (
        data.assign(half_hour=pd.to_datetime(data["order_time"]).dt.floor("30min").dt.strftime("%H:%M"))
        .groupby(["window", "period"], as_index=False)
        .agg(orders=("order_id", "nunique"), sales=("amount", "sum"))
        .sort_values(["orders", "sales"], ascending=False)
    )

    student_report = student_budget_report(orders, demo_student, 600.0)
    nutrition = nutrition_structure(orders, demo_student, recent_days=7)
    combo_suggestions = build_combo_suggestions(rules, dish_sales)
    lifecycle = build_lifecycle(data)

    feedback = _load_optional_csv(FEEDBACK_PATH)
    preferences = _load_optional_csv(PREFERENCES_PATH)
    votes = _load_optional_csv(VOTES_PATH)
    announcements = _load_optional_csv(ANNOUNCEMENTS_PATH)
    if announcements.empty:
        announcements = _default_announcements()

    feedback_insights = build_feedback_insights(feedback)
    comment_analysis = build_comment_keyword_analysis(feedback, data)
    promo_evaluation = build_promo_evaluation(data, promotions)
    resource_utilization = build_resource_utilization(canteen_compare, meal_plan)
    satisfaction_trend = build_satisfaction_trend(data, feedback)
    efficiency_ranking = build_canteen_efficiency(canteen_compare, resource_utilization)
    risk_overview = build_risk_overview(meal_plan)
    category_structure = build_category_structure(data)
    decision_actions = build_decision_actions(efficiency_ranking, risk_overview, feedback_insights, category_structure)
    report_sections = build_report_sections(daily_sales, efficiency_ranking, risk_overview)
    alert_center, alert_summary = build_alert_center(data, meal_plan)
    strategy_baseline = build_strategy_baseline(canteen_compare, risk_overview, crowding, efficiency_ranking)

    return {
        "meta": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "student_ids": student_ids[:200],
            "demo_student": demo_student,
            "canteens": sorted(data["canteen"].dropna().astype(str).unique().tolist()),
            "periods": ["早餐", "午餐", "晚餐", "夜宵"],
        },
        "summary": {
            "kpis": kpis,
            "quality": quality,
        },
        "student": {
            "todayRecommendation": json_ready_records(peer_recommendation),
            "peerRecommendation": json_ready_records(peer_recommendation),
            "budgetReport": {
                key: (value if not isinstance(value, pd.DataFrame) else json_ready_records(value))
                for key, value in student_report.items()
            },
            "nutrition": json_ready_records(nutrition),
            "crowding": json_ready_records(crowding),
            "preferences": json_ready_records(preferences),
            "feedback": json_ready_records(feedback),
            "votes": json_ready_records(votes),
            "announcements": json_ready_records(announcements),
        },
        "management": {
            "dailySales": json_ready_records(daily_sales),
            "canteenCompare": json_ready_records(canteen_compare),
            "dishSales": json_ready_records(dish_sales, limit=30),
            "hotDishes": json_ready_records(hot_dishes),
            "promotions": json_ready_records(promotions, limit=20),
            "lifecycle": json_ready_records(lifecycle, limit=30),
            "periodStat": json_ready_records(period_stat),
            "heatmap": json_ready_records(heat),
            "halfHourOrders": json_ready_records(half_hour),
            "windowPressure": json_ready_records(window_pressure, limit=30),
            "rules": json_ready_records(rules, limit=60),
            "comboSuggestions": json_ready_records(combo_suggestions, limit=12),
            "predictionMetrics": model_pack["metrics"],
            "predictionEval": json_ready_records(model_pack["eval_df"], limit=120),
            "futurePrediction": json_ready_records(future, limit=80),
            "mealPlan": json_ready_records(meal_plan, limit=80),
            "feedbackInsights": feedback_insights,
            "commentAnalysis": comment_analysis,
            "promoEvaluation": json_ready_records(promo_evaluation),
            "alertCenter": json_ready_records(alert_center),
            "alertSummary": alert_summary,
        },
        "logistics": {
            "canteenCompare": json_ready_records(canteen_compare),
            "canteenDailySales": json_ready_records(canteen_daily_sales),
            "clusterProfile": json_ready_records(cluster_profile),
            "studentFeatures": json_ready_records(student_features),
            "mealPlan": json_ready_records(meal_plan, limit=80),
            "resourceUtilization": json_ready_records(resource_utilization),
            "feedbackInsights": feedback_insights,
            "satisfactionTrend": json_ready_records(satisfaction_trend),
            "efficiencyRanking": json_ready_records(efficiency_ranking),
            "riskOverview": json_ready_records(risk_overview),
            "categoryStructure": json_ready_records(category_structure),
            "decisionActions": json_ready_records(decision_actions),
            "reportSections": report_sections,
            "alertCenter": json_ready_records(alert_center),
            "alertSummary": alert_summary,
            "strategySimulator": json_ready_records(strategy_baseline),
        },
        "raw": {
            "orders": json_ready_records(data),
        },
    }


def export_dashboard_data() -> Path:
    FRONTEND_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dashboard_payload()
    FRONTEND_DATA_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return FRONTEND_DATA_PATH


if __name__ == "__main__":
    print(export_dashboard_data())
