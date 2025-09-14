import streamlit as st
import pandas as pd
from datetime import date
import os

# 保存先ファイル
CSV_FILE = "experiment_notes.csv"

# 初期化：CSVがなければ作る
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["日付", "実験タイトル", "条件", "結果", "考察"])
    df.to_csv(CSV_FILE, index=False, encoding="utf-8")

# ページ設定
st.set_page_config(page_title="のーとAI（手入力版）", layout="centered")
st.title("📝 のーとAI（手入力版）")

# 入力フォーム
with st.form("note_form"):
    title = st.text_input("🧪 実験タイトル")
    condition = st.text_area("⚙️ 条件（例: 温度120℃, 時間30分）")
    result = st.text_area("📊 結果（例: CL強度50）")
    discussion = st.text_area("💡 考察")

    submitted = st.form_submit_button("保存")

    if submitted:
        new_note = pd.DataFrame([{
            "日付": str(date.today()),
            "実験タイトル": title,
            "条件": condition,
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
