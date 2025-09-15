import streamlit as st
import pandas as pd
from datetime import date, datetime
import os
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
import seaborn as sns

# 保存先ファイル
NOTE_FILE = "notes.csv"
CSV_FILE = "experiment_data.csv"

# 初期化
if not os.path.exists(NOTE_FILE):
    df = pd.DataFrame(columns=["実験ID", "日付", "目的", "結果", "考察"])
    df.to_csv(NOTE_FILE, index=False, encoding="utf-8")

if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["実験ID", "条件1", "条件2", "条件3", "結果"])
    df.to_csv(CSV_FILE, index=False, encoding="utf-8")

# ページ設定
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
        condition1 = st.number_input("⚙️ 条件1", step=1.0, format="%.2f")
        condition2 = st.number_input("⚙️ 条件2", step=1.0, format="%.2f")
        condition3 = st.number_input("⚙️ 条件3", step=1.0, format="%.2f")
        result_val = st.number_input("📊 結果", step=1.0, format="%.2f")

        submitted_csv = st.form_submit_button("CSVに保存")

        if submitted_csv:
            data_df = pd.read_csv(CSV_FILE)
            exp_id = f"{date.today().strftime('%Y%m%d')}-{len(data_df)+1:02d}"
            new_data = pd.DataFrame([{
                "実験ID": exp_id,
                "条件1": condition1,
                "条件2": condition2,
                "条件3": condition3,
                "結果": result_val
            }])
            data_df = pd.concat([data_df, new_data], ignore_index=True)
            data_df.to_csv(CSV_FILE, index=False, encoding="utf-8")
            st.success(f"✅ 条件と結果を保存しました！（ID: {exp_id}）")

    st.subheader("📑 実験データ (CSV)")
    data = pd.read_csv(CSV_FILE)
    st.dataframe(data)

    # --- グラフ描画 ---
    if not data.empty:
        fig, ax = plt.subplots()
        ax.scatter(data["条件1"], data["結果"], c="blue")
        ax.set_xlabel("条件1")
        ax.set_ylabel("結果")
        ax.set_title("条件1 vs 結果")
        st.pyplot(fig)

# =========================
# 📊 解析 (ベイズ最適化付き)
# =========================
with tab2:
    st.subheader("✨ 次の実験条件を提案します")

    df = pd.read_csv(CSV_FILE)

    mode = st.radio("最適化の目的を選択", ["最大化", "最小化"])

    if df.empty or len(df) < 3:
        st.warning("📉 データ点数が少ないです。もう少しデータを追加してください。")
    else:
        if st.button("🚀 解析スタート"):
            X = df[["条件1", "条件2", "条件3"]].values.tolist()
            y = df["結果"].tolist()

            if mode == "最大化":
                y = [-val for val in y]  # gp_minimizeは最小化なので符号反転

            space = [
                Real(min(df["条件1"]), max(df["条件1"]), name="条件1"),
                Real(min(df["条件2"]), max(df["条件2"]), name="条件2"),
                Real(min(df["条件3"]), max(df["条件3"]), name="条件3")
            ]

            res = gp_minimize(
                lambda x: None,  # 予測モデルで評価するのでダミー
                space,
                x0=X,
                y0=y,
                n_calls=max(len(X)+5, 20),
                random_state=42
            )

            proposed = res.x
            st.success(f"🔮 提案された次の条件: 条件1={proposed[0]:.2f}, 条件2={proposed[1]:.2f}, 条件3={proposed[2]:.2f}")

            # 可視化モード選択
            viz_mode = st.radio("可視化方法", ["散布図", "ヒートマップ（条件1 vs 条件2）", "履歴曲線"])

            if viz_mode == "散布図":
                fig, ax = plt.subplots()
                ax.scatter(df["条件1"], df["結果"], c="blue", label="実験データ")
                ax.set_xlabel("条件1")
                ax.set_ylabel("結果")
                ax.set_title("条件1 vs 結果")
                st.pyplot(fig)

            elif viz_mode == "ヒートマップ（条件1 vs 条件2）":
                if len(df) > 5:
                    pivot_df = df.pivot_table(index="条件1", columns="条件2", values="結果", aggfunc="mean")
                    fig, ax = plt.subplots()
                    sns.heatmap(pivot_df, cmap="viridis", ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("📉 データ点数が少ないため、ヒートマップを生成できません。")

            elif viz_mode == "履歴曲線":
                fig, ax = plt.subplots()
                ax.plot(range(1, len(df)+1), df["結果"], marker="o")
                ax.set_xlabel("試行回数")
                ax.set_ylabel("結果")
                ax.set_title("最適化履歴")
                st.pyplot(fig)

# =========================
# 🔄 データ共有・同期機能（簡易）
# =========================
st.sidebar.subheader("データ共有・アップロード")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード")
if uploaded_file:
    new_df = pd.read_csv(uploaded_file)
    new_df.to_csv(CSV_FILE, index=False, encoding="utf-8")
    st.sidebar.success("✅ データをアップロードしました")

uploaded_img = st.sidebar.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"])
if uploaded_img:
    st.sidebar.image(uploaded_img, caption="アップロード画像", use_column_width=True)
