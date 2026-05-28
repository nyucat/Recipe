from __future__ import annotations

import pandas as pd


def build_feature_importance_report(model_pack: dict) -> dict:
    model = model_pack["model"]
    feature_cols = model_pack["feature_cols"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return {"topFeatures": [], "groupedImportance": [], "insights": []}

    df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
    top_features = df.head(12).copy()
    top_features["feature_label"] = top_features["feature"].apply(_feature_label)

    grouped = (
        df.assign(feature_group=df["feature"].apply(_feature_group))
        .groupby("feature_group", as_index=False)
        .agg(importance=("importance", "sum"))
        .sort_values("importance", ascending=False)
    )
    grouped["importance_pct"] = grouped["importance"] / grouped["importance"].sum()

    insights = _build_insights(grouped, top_features)
    return {
        "topFeatures": top_features[["feature", "feature_label", "importance"]].to_dict(orient="records"),
        "groupedImportance": grouped.to_dict(orient="records"),
        "insights": insights,
    }


def _feature_group(name: str) -> str:
    if name in {"lag_1", "lag_7", "rolling_mean_7"}:
        return "历史销量特征"
    if name == "is_weekend" or name.startswith("weekday_"):
        return "时间特征"
    if name.startswith("dish_name_") or name.startswith("category_"):
        return "菜品特征"
    if name.startswith("canteen_") or name.startswith("window_"):
        return "食堂窗口特征"
    return "其他特征"


def _feature_label(name: str) -> str:
    mapping = {
        "lag_1": "前 1 天销量",
        "lag_7": "前 7 天销量",
        "rolling_mean_7": "过去 7 天平均销量",
        "is_weekend": "是否周末",
    }
    if name in mapping:
        return mapping[name]
    if name.startswith("weekday_"):
        return f"星期 {name.split('_', 1)[1]}"
    if name.startswith("dish_name_"):
        return f"菜品：{name.split('_', 1)[1]}"
    if name.startswith("category_"):
        return f"类别：{name.split('_', 1)[1]}"
    if name.startswith("canteen_"):
        return f"食堂：{name.split('_', 1)[1]}"
    if name.startswith("window_"):
        return f"窗口：{name.split('_', 1)[1]}"
    return name


def _build_insights(grouped: pd.DataFrame, top_features: pd.DataFrame) -> list[str]:
    rows = grouped.to_dict(orient="records")
    if not rows:
        return []
    top_group = rows[0]
    top_feature = top_features.iloc[0]["feature_label"] if not top_features.empty else "暂无"
    insights = [
        f"影响预测结果最大的特征组是“{top_group['feature_group']}”，贡献度约为 {top_group['importance_pct']:.0%}。",
        f"单个最重要的特征是“{top_feature}”，说明模型高度依赖近期历史销量和具体菜品行为。",
    ]
    group_names = {item["feature_group"]: item["importance_pct"] for item in rows}
    if group_names.get("时间特征", 0) >= 0.12:
        insights.append("时间特征占比不低，说明工作日/周末与星期差异会显著影响销量。")
    if group_names.get("食堂窗口特征", 0) >= 0.12:
        insights.append("食堂与窗口差异具有明显影响，说明不同供应场景存在稳定消费偏好。")
    if group_names.get("菜品特征", 0) >= 0.12:
        insights.append("菜品与品类特征占比较高，说明不同菜品本身的吸引力差异明显。")
    return insights[:4]
