import streamlit as st
import pandas as pd
from datetime import date
import os
from skopt import gp_minimize
from skopt.space import Real

# 保存先ファイル
CSV_FILE = "experiment_notes.csv"

# 初期化：CSVがなければ作る
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["日付", "温度", "時間", "結果", "考察"])
    df.to_csv(CSV_FILE, index=False, encoding="utf-8")

# ページ設定
st.set_page_config(page_title="のーとAI（ベイズ最適化版）", layout="centered")
st.title("📝 のーとAI（ベイズ最適化付き）")

# 入力フォーム
with st.form("note_form"):
    temp = st.number_input("🌡️ 温度 (50〜200℃)", min_value=50, max_value=200, step=1)
    time = st.number_input("⏱️ 時間 (10〜120分)", min_value=10, max_value=120, step=1)
    result = st.number_input("📊 結果スコア", step=1.0, format="%.2f")
    discussion = st.text_area("💡 考察")

    submitted = st.form_submit_button("保存")

    if submitted:
        new_note = pd.DataFrame([{
            "日付": str(date.today()),
            "温度": temp,
            "時間": time,
            "結果": result,
            "考察": discussion
        }])

        df = pd.read_csv(CSV_FILE)
        df = pd.concat([df, new_note], ignore_index=True)
        df.to_csv(CSV_FILE, index=False, encoding="utf-8")
        st.success("✅ 実験ノートを保存しました！")

# 保存されたノート一覧
st.subheader("📒 実験ノート一覧")
df = pd.read_csv(CSV_FILE)
st.dataframe(df)

# ベイズ最適化で次の条件提案
if len(df) > 2:  # データが最低3件以上あれば最適化を実行
    st.subheader("🔮 次の実験条件を提案します")

    X = df[["温度", "時間"]].values.tolist()
    y = df["結果"].values.tolist()

    # 探索範囲を定義
    space = [Real(50, 200, name="温度"), Real(10, 120, name="時間")]

    # ダミー目的関数（結果の最小化を仮定）
    def objective(x):
        return 0.0  # 実際の実験結果は未知なので空関数でOK（データを使って初期化する）

    # 既存データを使ってベイズ最適化
    res = gp_minimize(objective, space, x0=X, y0=y, n_calls=len(X) + 1, random_state=42)

    next_point = res.x_iters[-1]
    st.write(f"👉 推奨条件: 温度 **{next_point[0]:.1f}℃**, 時間 **{next_point[1]:.1f}分**")
