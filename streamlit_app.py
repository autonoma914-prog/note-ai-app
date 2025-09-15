import streamlit as st
import pandas as pd
from datetime import date
import os
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
import numpy as np

# 保存先ファイル
NOTE_FILE = "notes.csv"
CSV_FILE = "experiment_data.csv"

# =========================
# 動的な条件数の管理
# =========================
if "num_conditions" not in st.session_state:
    st.session_state.num_conditions = 1  # 最初は条件1だけ

# =========================
# 初期化とフォーマット修正
# =========================
if not os.path.exists(NOTE_FILE):
    df = pd.DataFrame(columns=["実験ID", "日付", "目的", "結果", "考察"])
    df.to_csv(NOTE_FILE, index=False, encoding="utf-8")

if not os.path.exists(CSV_FILE):
    cols = ["実験ID"] + [f"条件{i+1}" for i in range(st.session_state.num_conditions)] + ["結果"]
    df = pd.DataFrame(columns=cols)
    df.to_csv(CSV_FILE, index=False, encoding="utf-8")

# =========================
# ページ設定
# =========================
st.set_page_config(page_title="Autonoma", layout="centered")
st.title("Autonoma")

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
            notes_df = pd.read_csv(NOTE_FILE)
            exp_id = f"{date.today().strftime('%Y%m%d')}-{len(notes_df)+1:02d}"
            new_note = pd.DataFrame([{
                "実験ID": exp_id,
                "日付": str(date.today()),
                "目的": purpose,
                "結果": result_text,
                "考察": discussion
            }])
            notes_df = pd.concat([notes_df, new_note], ignore_index=True)
            notes_df.to_csv(NOTE_FILE, index=False, encoding="utf-8")
            st.success(f"✅ 実験ノートを保存しました！（ID: {exp_id}）")

    st.subheader("📒 実験ノート一覧")
    notes = pd.read_csv(NOTE_FILE)
    st.dataframe(notes)

    # --- 条件と結果を記入 ---
    st.subheader("条件と結果を記入 (CSV用)")

    with st.form("csv_form"):
        conditions = []
        # 最初は条件1のみを表示、追加されたらその数だけ表示
        for i in range(st.session_state.num_conditions):
            val = st.number_input(f"⚙️ 条件{i+1}", step=1.0, format="%.2f", key=f"cond_{i}")
            conditions.append(val)

        add_condition = st.form_submit_button("＋ 条件を追加")
        if add_condition:
            st.session_state.num_conditions += 1
            st.experimental_rerun()

        result_val = st.number_input("📊 結果", step=1.0, format="%.2f")

        submitted_csv = st.form_submit_button("CSVに保存")

        if submitted_csv:
            data_df = pd.read_csv(CSV_FILE)
            exp_id = f"{date.today().strftime('%Y%m%d')}-{len(data_df)+1:02d}"
            row = {"実験ID": exp_id}
            for i, val in enumerate(conditions):
                row[f"条件{i+1}"] = val
            row["結果"] = result_val

            new_data = pd.DataFrame([row])

            # 足りない列を埋める
            for col in row.keys():
                if col not in data_df.columns:
                    data_df[col] = np.nan

            data_df = pd.concat([data_df, new_data], ignore_index=True)
            data_df.to_csv(CSV_FILE, index=False, encoding="utf-8")
            st.success(f"✅ 条件と結果を保存しました！（ID: {exp_id}）")

    st.subheader("📑 実験データ (CSV)")
    data = pd.read_csv(CSV_FILE)
    st.dataframe(data)

# =========================
# 📊 解析 (ベイズ最適化付き)
# =========================
with tab2:
    st.subheader("✨ 次の実験条件を提案します")

    df = pd.read_csv(CSV_FILE)

    mode = st.radio("最適化の目的を選択", ["最大化", "最小化"])

    condition_cols = [col for col in df.columns if col.startswith("条件")]

    if df.empty or len(df) < 3:
        st.warning("📉 データ点数が少ないです。もう少しデータを追加してください。")
    else:
        if st.button("🚀 解析スタート"):
            X = df[condition_cols].values.tolist()
            y = df["結果"].tolist()

            if mode == "最大化":
                y = [-val for val in y]

            space = [Real(min(df[col]), max(df[col]), name=col) for col in condition_cols]

            res = gp_minimize(
                lambda x: None,
                space,
                x0=X,
                y0=y,
                n_calls=max(len(X)+5, 20),
                random_state=42
            )

            proposed = res.x
            st.success("🔮 提案された次の条件:" + ", ".join([f"{col}={val:.2f}" for col, val in zip(condition_cols, proposed)]))

            # 可視化モード選択
            viz_mode = st.radio("可視化方法", ["散布図", "履歴曲線"])

            if viz_mode == "散布図" and condition_cols:
                fig, ax = plt.subplots()
                ax.scatter(df[condition_cols[0]], df["結果"], c="blue", label="実験データ")
                ax.set_xlabel(condition_cols[0])
                ax.set_ylabel("結果")
                ax.set_title(f"{condition_cols[0]} vs 結果")
                st.pyplot(fig)

            elif viz_mode == "履歴曲線":
                fig, ax = plt.subplots()
                ax.plot(range(1, len(df)+1), df["結果"], marker="o")
                ax.set_xlabel("試行回数")
                ax.set_ylabel("結果")
                ax.set_title("最適化履歴")
                st.pyplot(fig)
