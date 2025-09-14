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

# サイドバーでページ選択
page = st.sidebar.radio("📑 目次", ["ノート", "解析"])

# ==========================================================
# 📒 実験ノートページ
# ==========================================================
if page == "ノート":
    st.title("📝 実験ノート - Autonoma")

    # 実験入力フォーム
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

    # 保存されたノート一覧
    st.subheader("📒 実験ノート一覧")
    df = pd.read_csv(CSV_FILE)
    st.dataframe(df)

    # ノート削除
    if not df.empty:
        st.subheader("🗑️ ノート削除")
        delete_index = st.number_input("削除したい行番号を指定してください", min_value=0, max_value=len(df)-1, step=1)
        if st.button("削除"):
            df = df.drop(delete_index).reset_index(drop=True)
            df.to_csv(CSV_FILE, index=False, encoding="utf-8")
            st.success(f"行 {delete_index} を削除しました。")
            st.experimental_rerun()

# ==========================================================
# 🔮 解析ページ（ベイズ最適化）
# ==========================================================
elif page == "解析":
    st.title("🔮 解析 - Autonoma")

    df = pd.read_csv(CSV_FILE)
    st.subheader("📊 現在のデータ")
    st.dataframe(df)

    # ベイズ最適化を実行
    st.subheader("✨ 次の実験条件を提案")
    if len(df) >= 3:
        X = df[["条件"]].values.tolist()
        y = df["結果"].values.tolist()

        def objective(x):
            idx = [row[0] for row in X].index(x[0]) if x[0] in [row[0] for row in X] else -1
            if idx >= 0:
                return y[idx]
            else:
                return 0

        space = [Real(0.0, 1000.0, name="condition")]

        res = gp_minimize(
            objective,
            space,
            x0=X,
            y0=y,
            n_calls=len(X) + 5,
            random_state=42
        )

        st.success(f"🧪 推奨される次の条件: {res.x[0]:.2f}")
    else:
        st.info("⚠️ ベイズ最適化には少なくとも3件以上のデータが必要です。")


