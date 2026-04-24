from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "student_id",
    "order_id",
    "order_time",
    "canteen",
    "window",
    "dish_name",
    "category",
    "price",
    "quantity",
    "amount",
    "payment",
    "rating",
]


@dataclass(frozen=True)
class Dish:
    name: str
    category: str
    price: float
    popularity: float
    windows: tuple[str, ...]
    periods: tuple[str, ...]


DISHES = [
    Dish("包子", "主食", 2.5, 0.95, ("早餐窗口",), ("早餐",)),
    Dish("豆浆", "饮品", 2.0, 0.92, ("早餐窗口",), ("早餐",)),
    Dish("鸡蛋", "主食", 1.8, 0.88, ("早餐窗口",), ("早餐",)),
    Dish("热干面", "主食", 8.0, 0.86, ("面食窗口",), ("早餐", "午餐")),
    Dish("牛肉面", "主食", 14.0, 0.83, ("面食窗口",), ("午餐", "晚餐")),
    Dish("黄焖鸡米饭", "主食", 16.0, 0.9, ("盖饭窗口",), ("午餐", "晚餐")),
    Dish("麻辣香锅", "主食", 18.0, 0.87, ("特色窗口",), ("午餐", "晚餐")),
    Dish("番茄炒蛋", "热菜", 10.0, 0.84, ("快餐窗口",), ("午餐", "晚餐")),
    Dish("红烧肉", "热菜", 14.0, 0.85, ("快餐窗口",), ("午餐", "晚餐")),
    Dish("青椒土豆丝", "热菜", 8.0, 0.78, ("快餐窗口",), ("午餐", "晚餐")),
    Dish("紫菜蛋花汤", "汤品", 4.0, 0.7, ("汤品窗口",), ("午餐", "晚餐")),
    Dish("绿豆汤", "汤品", 4.5, 0.74, ("汤品窗口",), ("午餐", "晚餐", "夜宵")),
    Dish("奶茶", "饮品", 10.0, 0.82, ("饮品窗口",), ("午餐", "晚餐", "夜宵")),
    Dish("可乐", "饮品", 4.0, 0.68, ("饮品窗口",), ("午餐", "晚餐", "夜宵")),
    Dish("炸鸡排", "小吃", 11.0, 0.76, ("小吃窗口",), ("晚餐", "夜宵")),
    Dish("炒饭", "主食", 12.0, 0.8, ("夜宵窗口",), ("夜宵",)),
    Dish("炒面", "主食", 12.5, 0.79, ("夜宵窗口",), ("夜宵",)),
    Dish("水果沙拉", "轻食", 12.0, 0.55, ("轻食窗口",), ("午餐", "晚餐")),
]

CANTEENS = ["一食堂", "二食堂", "三食堂"]
WINDOWS = [
    "早餐窗口",
    "快餐窗口",
    "盖饭窗口",
    "面食窗口",
    "汤品窗口",
    "饮品窗口",
    "小吃窗口",
    "夜宵窗口",
    "轻食窗口",
    "特色窗口",
]
PAYMENTS = ["校园卡", "微信", "支付宝"]


def _period_from_hour(hour: int) -> str:
    if 6 <= hour < 10:
        return "早餐"
    if 10 <= hour < 14:
        return "午餐"
    if 16 <= hour < 20:
        return "晚餐"
    return "夜宵"


def generate_sample_orders(
    n_orders: int = 4000,
    n_students: int = 500,
    start_date: str = "2026-02-01",
    n_days: int = 75,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime.fromisoformat(start_date)
    students = [f"S{str(i).zfill(4)}" for i in range(1, n_students + 1)]
    records: list[dict] = []

    meal_hour_options = {
        "早餐": np.array([6, 7, 8, 9]),
        "午餐": np.array([10, 11, 12, 13]),
        "晚餐": np.array([16, 17, 18, 19]),
        "夜宵": np.array([20, 21, 22]),
    }
    meal_prob = np.array([0.26, 0.38, 0.27, 0.09])
    canteen_prob = np.array([0.42, 0.33, 0.25])

    for idx in range(1, n_orders + 1):
        meal = rng.choice(["早餐", "午餐", "晚餐", "夜宵"], p=meal_prob)
        hour = int(rng.choice(meal_hour_options[meal]))
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        day_offset = int(rng.integers(0, n_days))
        order_time = start + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)

        canteen = rng.choice(CANTEENS, p=canteen_prob)
        student = rng.choice(students)
        payment = rng.choice(PAYMENTS, p=[0.5, 0.28, 0.22])
        item_count = int(rng.choice([1, 2, 3], p=[0.58, 0.33, 0.09]))
        order_id = f"O{order_time.strftime('%Y%m%d')}{idx:05d}"

        candidate = [d for d in DISHES if meal in d.periods]
        pop = np.array([d.popularity for d in candidate], dtype=float)
        pop = pop / pop.sum()

        chosen = rng.choice(candidate, size=item_count, replace=False, p=pop)
        for dish in chosen:
            quantity = int(rng.choice([1, 1, 1, 2], p=[0.4, 0.3, 0.2, 0.1]))
            price_noise = float(rng.normal(0, 0.4))
            price = round(max(1.0, dish.price + price_noise), 1)
            amount = round(price * quantity, 2)
            rating = float(np.clip(rng.normal(3.8 + 0.7 * (dish.popularity - 0.5), 0.45), 2.0, 5.0))
            window = dish.windows[0] if dish.windows else rng.choice(WINDOWS)

            records.append(
                {
                    "student_id": student,
                    "order_id": order_id,
                    "order_time": order_time,
                    "canteen": canteen,
                    "window": window,
                    "dish_name": dish.name,
                    "category": dish.category,
                    "price": price,
                    "quantity": quantity,
                    "amount": amount,
                    "payment": payment,
                    "rating": round(rating, 1),
                }
            )

    return pd.DataFrame(records, columns=EXPECTED_COLUMNS)


def detect_data_quality(df: pd.DataFrame) -> dict[str, int]:
    check = df.copy()
    missing_values = int(check.isna().sum().sum())
    duplicate_lines = int(check.duplicated().sum())
    invalid_price = int((pd.to_numeric(check.get("price", pd.Series(dtype=float)), errors="coerce") <= 0).sum())
    invalid_quantity = int(
        (pd.to_numeric(check.get("quantity", pd.Series(dtype=float)), errors="coerce") <= 0).sum()
    )
    invalid_time = int(pd.to_datetime(check.get("order_time", pd.Series(dtype=str)), errors="coerce").isna().sum())
    return {
        "missing_values": missing_values,
        "duplicate_lines": duplicate_lines,
        "invalid_price": invalid_price,
        "invalid_quantity": invalid_quantity,
        "invalid_time": invalid_time,
    }


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少字段: {', '.join(missing_cols)}")

    clean = df.copy()
    clean["order_time"] = pd.to_datetime(clean["order_time"], errors="coerce")

    for col in ["student_id", "order_id", "canteen", "window", "dish_name", "category", "payment"]:
        clean[col] = clean[col].astype(str).str.strip()

    clean["price"] = pd.to_numeric(clean["price"], errors="coerce")
    clean["quantity"] = pd.to_numeric(clean["quantity"], errors="coerce")
    clean["rating"] = pd.to_numeric(clean["rating"], errors="coerce")
    clean["rating"] = clean["rating"].fillna(clean["rating"].median()).clip(1, 5)

    clean = clean.dropna(subset=["order_time", "student_id", "order_id", "dish_name", "price", "quantity"])
    clean = clean[(clean["price"] > 0) & (clean["price"] <= 80)]
    clean = clean[(clean["quantity"] > 0) & (clean["quantity"] <= 5)]
    clean["quantity"] = clean["quantity"].astype(int)
    clean["dish_name"] = clean["dish_name"].str.replace(r"\s+", "", regex=True)
    clean = clean.drop_duplicates(subset=["order_id", "dish_name"], keep="first")
    clean["amount"] = (clean["price"] * clean["quantity"]).round(2)
    clean = clean.sort_values("order_time").reset_index(drop=True)
    return clean[EXPECTED_COLUMNS]


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
