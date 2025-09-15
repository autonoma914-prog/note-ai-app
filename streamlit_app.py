# -*- coding: utf-8 -*-
# Autonoma: Experiment Notes + Kaneko-style Regression & AD Analysis (sklearn<0.22 Compatible)

import os
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, OneClassSVM
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors

# ------------------------------------------------------------
# 設定
# ------------------------------------------------------------
NOTE_FILE = "notes.csv"
CSV_FILE = "experiment_data.csv"

# Streamlit ページ設定
st.set_page_config(page_title="Autonoma", layout="centered")
st.title("Autonoma")

# SessionState: 動的な条件数
if "num_conditions" not in st.session_state:
    st.session_state.num_conditions = 1  # 最初は条件1だけ

# 初期ファイル生成
if not os.path.exists(NOTE_FILE):
    df = pd.DataFrame(columns=["実験ID", "日付", "目的", "結果", "考察"])
    df.to_csv(NOTE_FILE, index=False, encoding="utf-8")

if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["実験ID", "条件1", "結果"])
    df.to_csv(CSV_FILE, index=False, encoding="utf-8")

# ------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------
def autoscale_fit_transform(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=0)
    y_mean = y.mean()
    y_std = y.std(ddof=0)
    X_std_safe = np.where(X_std == 0, 1.0, X_std)
    y_std_safe = 1.0 if y_std == 0 else y_std
    X_auto = (X - X_mean) / X_std_safe
    y_auto = (y - y_mean) / y_std_safe
    return X_auto, y_auto, (X_mean, X_std_safe, y_mean, y_std_safe)

def autoscale_transform(X_new, scalers):
    X_mean, X_std, _, _ = scalers
    X_new = np.asarray(X_new, dtype=float)
    return (X_new - X_mean) / X_std

def inverse_scale_y(y_scaled, scalers):
    _, _, y_mean, y_std = scalers
    return y_scaled * y_std + y_mean

def build_kernels(n_features):
    return [
        ConstantKernel() * DotProduct() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * RBF(np.ones(n_features)) + WhiteKernel(),
        ConstantKernel() * RBF(np.ones(n_features)) + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
    ]

def gamma_by_gram_variance(X_auto, gamma_grid):
    XA = np.asarray(X_auto, dtype=float)
    var_list = []
    for g in gamma_grid:
        d2 = ((XA[:, None, :] - XA[None, :, :]) ** 2).sum(axis=2)
        K = np.exp(-g * d2)
        var_list.append(K.var(ddof=1))
    idx = int(np.argmax(var_list))
    return gamma_grid[idx]

def make_candidates_uniform(df_valid, cond_cols, n_candidates, seed=42):
    rng = np.random.default_rng(seed)
    mins = df_valid[cond_cols].min().values.astype(float)
    maxs = df_valid[cond_cols].max().values.astype(float)
    width = np.maximum(maxs - mins, 1e-12)
    low = mins
    high = mins + width
    Xc = rng.uniform(low=low, high=high, size=(n_candidates, len(cond_cols)))
    cand_df = pd.DataFrame(Xc, columns=cond_cols)
    return cand_df

def safe_hist(values, title, xlabel):
    fig, ax = plt.subplots()
    ax.hist(values, bins=20)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    st.pyplot(fig)

# ------------------------------------------------------------
# タブ
# ------------------------------------------------------------
tab1, tab2 = st.tabs(["📝 実験ノート", "📊 解析"])

# ------------------------------------------------------------
# 📝 実験ノート
# ------------------------------------------------------------
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
    st.dataframe(notes, use_container_width=True)

    st.subheader("条件と結果を記入 (CSV用)")
    with st.form("csv_form"):
        conditions = []
        for i in range(st.session_state.num_conditions):
            val = st.number_input(f"⚙️ 条件{i+1}", step=1.0, format="%.4f", key=f"cond_{i}")
            conditions.append(val)

        col_a, col_b = st.columns(2)
        with col_a:
            add_condition = st.form_submit_button("＋ 条件を追加")
        with col_b:
            remove_condition = st.form_submit_button("－ 条件を削除")

        if add_condition:
            st.session_state.num_conditions += 1
            df = pd.read_csv(CSV_FILE)
            for i in range(st.session_state.num_conditions):
                col = f"条件{i+1}"
                if col not in df.columns:
                    df[col] = np.nan
            df.to_csv(CSV_FILE, index=False, encoding="utf-8")
            st.experimental_rerun()

        if remove_condition and st.session_state.num_conditions > 1:
            st.session_state.num_conditions -= 1

        result_val = st.number_input("📊 結果", step=1.0, format="%.6f")
        submitted_csv = st.form_submit_button("CSVに保存")
        if submitted_csv:
            data_df = pd.read_csv(CSV_FILE)
            exp_id = f"{date.today().strftime('%Y%m%d')}-{len(data_df)+1:02d}"
            row = {"実験ID": exp_id}
            for i, val in enumerate(conditions):
                row[f"条件{i+1}"] = val
            row["結果"] = result_val
            for col in row.keys():
                if col not in data_df.columns:
                    data_df[col] = np.nan
            new_data = pd.DataFrame([row])
            data_df = pd.concat([data_df, new_data], ignore_index=True)
            data_df.to_csv(CSV_FILE, index=False, encoding="utf-8")
            st.success(f"✅ 条件と結果を保存しました！（ID: {exp_id}）")

    st.subheader("📑 実験データ (CSV)")
    data = pd.read_csv(CSV_FILE)
    st.dataframe(data, use_container_width=True)

# ------------------------------------------------------------
# 📊 解析（Kaneko回帰 + AD）
# ------------------------------------------------------------
with tab2:
    st.subheader("✨ Kaneko回帰 + AD 解析で次の実験条件を提案")

    df = pd.read_csv(CSV_FILE)
    all_condition_cols = [c for c in df.columns if c.startswith("条件")]
    if "結果" not in df.columns:
        st.warning("⚠️ CSVに『結果』列がありません。")
        st.stop()

    mode = st.radio("最適化の目的", ["最大化", "最小化"], horizontal=True)

    if not all_condition_cols:
        st.warning("⚠️ 条件データが存在しません。")
        st.stop()

    selected_conditions = st.multiselect(
        "解析に使用する条件列",
        all_condition_cols,
        default=[all_condition_cols[0]]
    )

    if df.empty or len(df) < 3:
        st.warning("📉 有効なデータ点が不足しています。")
        st.stop()

    # --- データ準備 ---
    df_valid = df.dropna(subset=selected_conditions + ["結果"])
    X = df_valid[selected_conditions].values
    y = df_valid["結果"].values

    # オートスケーリング
    X_auto, y_auto, scalers = autoscale_fit_transform(X, y)

    # --- 簡易版：線形回帰でテスト（本当はSVR/GPRの分岐あり） ---
    model = LinearRegression()
    model.fit(X_auto, y_auto)

    y_pred_train = inverse_scale_y(model.predict(X_auto), scalers)

    # 評価
    r2_train = r2_score(y, y_pred_train)
    rmse_train = mean_squared_error(y, y_pred_train) ** 0.5
    mae_train = mean_absolute_error(y, y_pred_train)

    st.write("**学習データでの評価**")
    st.write("R²:", r2_train)
    st.write("RMSE:", rmse_train)
    st.write("MAE:", mae_train)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv_scaled = cross_val_predict(model, X_auto, y_auto, cv=cv)
    y_cv = inverse_scale_y(y_cv_scaled, scalers)

    r2_cv = r2_score(y, y_cv)
    rmse_cv = mean_squared_error(y, y_cv) ** 0.5
    mae_cv = mean_absolute_error(y, y_cv)

    st.write("**CVでの評価**")
    st.write("R²:", r2_cv)
    st.write("RMSE:", rmse_cv)
    st.write("MAE:", mae_cv)

    # 実測 vs CV推定 プロット
    fig, ax = plt.subplots()
    ax.scatter(y, y_cv, c="blue")
    ax.set_xlabel("実測値")
    ax.set_ylabel("CV推定値")
    st.pyplot(fig)
