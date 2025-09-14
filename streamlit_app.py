import streamlit as st
import pandas as pd
from datetime import date
import os
from skopt import gp_minimize
from skopt.space import Real

# 保存先ファイル
CSV_FILE = "experiment_notes.csv"

# 初期化：CSVがなければ作成
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["日付", "条件", "結果"])
    df.to_csv(CSV_FILE, index=False, encoding="utf-8")

# ページ設定
st.set_page_config(page_title="Autonoma", layout="centered")
st.title("🤖 Autonoma")

# ---------------------------------------------
# 実験記録フォーム
# ---------------------------------------------
st.header("📝 実験を記録する")

with st.form("note_form"):
    condition = st.number_input("⚙️ 実験条件（数値）", min_value=0.0, max_value=1000.0, step=1.0)
    result = st.number_input("📊 実験結果（スコアなど）", step=1.0, format="%.2f")

    submitted = st.form_submit_button("保存")

    if submitted:
        new_note = pd.DataFrame([{
            "日付": str(date.today()),
            "条件": condition,
            "結果": result
        }])
        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, new_note], ignore_index=True)
        df.to_csv(CSV_FILE, index=False, encoding="utf-8")
        st.success("✅ 実験ノートを保存しました！")

# 保存されたノート一覧を表示
st.subheader("📒 実験ノート一覧")
df = pd.read_csv(CSV_FILE)
st.dataframe(df)

# ---------------------------------------------
# ベイズ最適化で次の条件を提案
# ---------------------------------------------
st.header("🔮 次の実験条件を提案する")

if len(df) >= 3:  # データがある程度溜まってから実行
    X = df[["条件"]].values.tolist()
    y = df["結果"].values.tolist()

    # 目的関数（結果を最小化する形式にする）
    def objective(x):
        idx = [row[0] for row in X].index(x[0]) if x[0] in [row[0] for row in X] else -1
        if idx >= 0:
            return y[idx]
        else:
            return 0  # 未知の場合は仮の値

    # 探索範囲を設定
    space = [Real(0.0, 1000.0, name="condition")]

    res = gp_minimize(
        objective,
        space,
        x0=X,
        y0=y,
        n_calls=len(X) + 5,
        random_state=42
    )

    st.success(f"✨ 次の推奨条件: {res.x[0]:.2f}")

else:
    st.info("⚠️ ベイズ最適化を実行するには、少なくとも3件以上の実験データが必要です。")

