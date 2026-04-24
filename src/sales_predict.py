from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _daily_dish(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["order_time"]).dt.date
    daily = (
        data.groupby(["date", "dish_name", "category", "canteen", "window"], as_index=False)
        .agg(quantity=("quantity", "sum"))
        .copy()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values(["dish_name", "date"]).reset_index(drop=True)


def _build_supervised(daily: pd.DataFrame) -> pd.DataFrame:
    x = daily.copy()
    x["weekday"] = x["date"].dt.weekday
    x["is_weekend"] = (x["weekday"] >= 5).astype(int)
    x["lag_1"] = x.groupby("dish_name")["quantity"].shift(1)
    x["lag_7"] = x.groupby("dish_name")["quantity"].shift(7)
    x["rolling_mean_7"] = x.groupby("dish_name")["quantity"].transform(lambda s: s.shift(1).rolling(7).mean())
    x = x.dropna().reset_index(drop=True)
    return x


def _to_feature_matrix(df: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    X = pd.get_dummies(
        df[["dish_name", "category", "canteen", "window", "weekday", "is_weekend", "lag_1", "lag_7", "rolling_mean_7"]],
        columns=["dish_name", "category", "canteen", "window", "weekday"],
        dtype=float,
    )
    if feature_cols is None:
        feature_cols = X.columns.tolist()
    X = X.reindex(columns=feature_cols, fill_value=0.0)
    return X, feature_cols


def train_sales_model(df: pd.DataFrame) -> dict:
    daily = _daily_dish(df)
    supervised = _build_supervised(daily)
    if len(supervised) < 50:
        raise ValueError("可用于训练的数据不足，请增加历史数据。")

    X, feature_cols = _to_feature_matrix(supervised)
    y = supervised["quantity"].astype(float)

    split_date = supervised["date"].quantile(0.8)
    train_mask = supervised["date"] <= split_date
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    eval_df = supervised.loc[~train_mask, ["date", "dish_name", "quantity"]].copy()
    if X_test.empty:
        X_train, X_test = X.iloc[:-20], X.iloc[-20:]
        y_train, y_test = y.iloc[:-20], y.iloc[-20:]
        eval_df = supervised.iloc[-20:][["date", "dish_name", "quantity"]].copy()

    model = RandomForestRegressor(
        n_estimators=260,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    eval_df["pred"] = pred
    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
    }
    return {
        "model": model,
        "feature_cols": feature_cols,
        "daily": daily,
        "supervised": supervised,
        "eval_df": eval_df,
        "metrics": metrics,
    }


def forecast_next_days(model_pack: dict, days: int = 7) -> pd.DataFrame:
    model = model_pack["model"]
    feature_cols = model_pack["feature_cols"]
    history = model_pack["daily"].copy()
    history = history.sort_values(["dish_name", "date"]).reset_index(drop=True)

    latest_date = history["date"].max()
    dish_meta = history.groupby("dish_name", as_index=False).agg(
        category=("category", "first"), canteen=("canteen", "first"), window=("window", "first")
    )

    values_map: dict[str, list[float]] = {
        k: v["quantity"].astype(float).tolist() for k, v in history.groupby("dish_name")
    }
    preds: list[dict] = []

    for step in range(1, days + 1):
        current_date = latest_date + timedelta(days=step)
        weekday = int(current_date.weekday())
        is_weekend = int(weekday >= 5)

        rows = []
        for _, meta in dish_meta.iterrows():
            dish = meta["dish_name"]
            series = values_map[dish]
            lag_1 = series[-1]
            lag_7 = series[-7] if len(series) >= 7 else float(np.mean(series))
            rolling_mean_7 = float(np.mean(series[-7:])) if len(series) >= 7 else float(np.mean(series))
            rows.append(
                {
                    "dish_name": dish,
                    "category": meta["category"],
                    "canteen": meta["canteen"],
                    "window": meta["window"],
                    "weekday": weekday,
                    "is_weekend": is_weekend,
                    "lag_1": lag_1,
                    "lag_7": lag_7,
                    "rolling_mean_7": rolling_mean_7,
                }
            )

        feat_df = pd.DataFrame(rows)
        X_pred, _ = _to_feature_matrix(feat_df, feature_cols=feature_cols)
        y_pred = model.predict(X_pred)
        y_pred = np.maximum(y_pred, 0.0)

        for i, row in feat_df.iterrows():
            qty = float(y_pred[i])
            values_map[row["dish_name"]].append(qty)
            preds.append(
                {
                    "date": current_date.date(),
                    "dish_name": row["dish_name"],
                    "category": row["category"],
                    "canteen": row["canteen"],
                    "window": row["window"],
                    "pred_quantity": round(qty, 1),
                }
            )

    return pd.DataFrame(preds)


def build_meal_plan(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df
    out = pred_df.copy()
    out["suggested_prep"] = (out["pred_quantity"] * 1.12).round().astype(int)
    high = out["pred_quantity"].quantile(0.75)
    low = out["pred_quantity"].quantile(0.25)
    out["risk"] = "正常"
    out.loc[out["pred_quantity"] >= high, "risk"] = "缺货风险较高"
    out.loc[out["pred_quantity"] <= low, "risk"] = "浪费风险较高"
    out["advice"] = "保持供应"
    out.loc[out["risk"] == "缺货风险较高", "advice"] = "增加备餐并提前备货"
    out.loc[out["risk"] == "浪费风险较高", "advice"] = "适度减量并考虑促销"
    return out.sort_values(["date", "pred_quantity"], ascending=[True, False]).reset_index(drop=True)
