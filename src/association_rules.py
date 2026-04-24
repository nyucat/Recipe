from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def mine_association_rules(
    df: pd.DataFrame,
    min_support: float = 0.02,
    min_confidence: float = 0.3,
    min_lift: float = 1.1,
    max_len: int = 3,
) -> pd.DataFrame:
    tx = df.groupby("order_id")["dish_name"].apply(lambda x: list(set(x))).tolist()
    if not tx:
        return pd.DataFrame()

    te = TransactionEncoder()
    matrix = te.fit(tx).transform(tx)
    basket = pd.DataFrame(matrix, columns=te.columns_)
    freq_itemsets = apriori(basket, min_support=min_support, use_colnames=True, max_len=max_len)
    if freq_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return pd.DataFrame()
    rules = rules[rules["lift"] >= min_lift].copy()
    if rules.empty:
        return pd.DataFrame()

    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: "、".join(sorted(list(x))))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: "、".join(sorted(list(x))))
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    show_cols = [
        "antecedents_str",
        "consequents_str",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
    ]
    return rules[show_cols]


def recommend_by_selected_dish(rules_df: pd.DataFrame, dish_name: str, top_n: int = 5) -> pd.DataFrame:
    if rules_df.empty:
        return pd.DataFrame()
    mask = rules_df["antecedents_str"].str.contains(dish_name, na=False)
    rec = rules_df[mask].copy()
    if rec.empty:
        return pd.DataFrame()
    rec = rec.sort_values(["lift", "confidence"], ascending=False).head(top_n).reset_index(drop=True)
    rec["reason"] = rec.apply(
        lambda r: f"买了 {r['antecedents_str']} 的用户常一起买 {r['consequents_str']} (置信度 {r['confidence']:.2f})",
        axis=1,
    )
    return rec
