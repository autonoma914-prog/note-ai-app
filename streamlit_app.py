import streamlit as st
import pandas as pd
from datetime import date
import os
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt

# 保存先ファイル
NOTE_FILE = "notes.csv"
CSV_FILE = "experiment_data.csv"

# 初期化：ファイルがなければ作る
if not os.path.exists(NOTE_FILE):
    df = pd.DataFrame(columns=["日付", "目的", "結果", "考察"])
    df.to_csv(NOTE_FILE, index=False, encoding="utf-8")

if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["条件", "結果"])
    df.to_csv(CSV_FILE, index=False, encoding="utf-8")

# ページ設定
st.set_page_config(page_title="Autonoma", layout="centered")
st.title("🤖 Autonoma")

# タブ分け
tab1, tab2 = st.tabs(["📝 実験ノート", "📊 解析"])

# =========================
# 📝 実験ノート
# =========================
with tab1:
    st.subheader("新しい実験ノートを追加")

    with st.form("note_form"):
        purpose = st.text_area("🎯 目的")
        result_text = st.text_area("📊 結果")
        discussion = st.text_area("💡 考察")

        submitted = st.form_submit_button("保存")

        if submitted:
            new_note = pd.DataFrame([{
                "日付": str(date.today()),
                "目的": purpose,
                "結果": result_text,
                "考察": discussion
            }])

            df = pd.read_csv(NOTE_FILE)
            df = pd.concat([df, new_note], ignore_index=True)
            df.to_csv(NOTE_FILE, index=False, encoding="utf-8")
            st.success("✅ 実験ノートを保存しました！")

    st.subheader("📒 実験ノート一覧")
    notes = pd.read_csv(NOTE_FILE)
    st.dataframe(notes)

    # ノート削除機能
    if not notes.empty:
        delete_index = st.number_input("削除するノートの番号を指定", min_value=0, max_value=len(notes)-1, step=1)
        if st.button("🗑️ 削除"):
            notes = notes.drop(delete_index).reset_index(drop=True)
            notes.to_csv(NOTE_FILE, index=False, encoding="utf-8")
            st.success("✅ 指定したノートを削除しました")

    st.subheader("条件と結果を記入 (CSV用)")
    with st.form("csv_form"):
        condition = st.number_input("⚙️ 条件", step=1.0, format="%.2f")
        result_val = st.number_input("📊 結果", step=1.0, format="%.2f")

        submitted_csv = st.form_submit_button("CSVに保存")

        if submitted_csv:
            new_data = pd.DataFrame([{"条件": condition, "結果": result_val}])
            df = pd.read_csv(CSV_FILE)
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv(CSV_FILE, index=False, encoding="utf-8")
            st.success("✅ 条件と結果を保存しました！")

    st.subheader("📑 実験データ (CSV)")
    data = pd.read_csv(CSV_FILE)
    st.dataframe(data)

# =========================
# 📊 解析 (ベイズ最適化付き)
# =========================
with tab2:
    st.subheader("✨ 次の実験条件を提案します")

    df = pd.read_csv(CSV_FILE)

    if df.empty or len(df) < 3:
        st.warning("📉 データ点数が少ないです。もう少しデータを追加してください。")
    else:
        X = df[["条件"]].values.tolist()
        y = df["結果"].values.tolist()

        # 目的関数（ベイズ最適化用）
        def objective(x):
            idx = min(range(len(X)), key=lambda i: abs(X[i][0] - x[0]))
            return y[idx]

        space = [Real(min(df["条件"]), max(df["条件"]), name="condition")]
        n_calls = max(len(X) + 5, 15)

        res = gp_minimize(
            objective,
            space,
            x0=X,
            y0=y,
            n_calls=n_calls,
            random_state=42
        )

        st.success(f"🔮 提案された次の条件: {res.x[0]:.2f}")

        # 散布図を表示
        fig, ax = plt.subplots()
        ax.scatter(df["条件"], df["結果"], c="blue", label="実験データ")
        ax.set_xlabel("条件")
        ax.set_ylabel("結果")
        ax.set_title("条件 vs 結果")
        ax.legend()
        st.pyplot(fig)
