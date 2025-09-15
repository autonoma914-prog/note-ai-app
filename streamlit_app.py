# -*- coding: utf-8 -*-
# Autonoma: Experiment Notes + Kaneko-style Regression & AD Analysis (Integrated Full Version)

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
    """X, y をオートスケーリング。平均・標準偏差と共に返す"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=0)
    y_mean = y.mean()
    y_std = y.std(ddof=0)

    # ゼロ割回避（std=0は1に）
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
    """Kaneko式の11カーネル定義"""
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
    """分散最大化で RBF gamma を選択"""
    XA = np.asarray(X_auto, dtype=float)
    var_list = []
    for g in gamma_grid:
        # RBF Gram matrix
        d2 = ((XA[:, None, :] - XA[None, :, :]) ** 2).sum(axis=2)
        K = np.exp(-g * d2)
        var_list.append(K.var(ddof=1))
    idx = int(np.argmax(var_list))
    return gamma_grid[idx]

def make_candidates_uniform(df_valid, cond_cols, n_candidates, seed=42):
    """min-max の一様サンプリングで候補集合を作る"""
    rng = np.random.default_rng(seed)
    mins = df_valid[cond_cols].min().values.astype(float)
    maxs = df_valid[cond_cols].max().values.astype(float)
    # 同一値範囲は僅かに広げる（数値安定）
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

    # --- 条件と結果を記入 ---
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
            # 既存CSVの列はそのまま保持（過去データ互換のため）

        result_val = st.number_input("📊 結果", step=1.0, format="%.6f")

        submitted_csv = st.form_submit_button("CSVに保存")

        if submitted_csv:
            data_df = pd.read_csv(CSV_FILE)
            exp_id = f"{date.today().strftime('%Y%m%d')}-{len(data_df)+1:02d}"
            row = {"実験ID": exp_id}
            for i, val in enumerate(conditions):
                row[f"条件{i+1}"] = val
            row["結果"] = result_val

            # 足りない列を埋める
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
        st.warning("⚠️ CSVに『結果』列がありません。まずデータを追加してください。")
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

    col_top1, col_top2, col_top3 = st.columns(3)
    with col_top1:
        regression_method = st.selectbox(
            "回帰手法",
            ["ols_linear", "ols_nonlinear", "svr_linear", "svr_gaussian", "gpr_one_kernel", "gpr_kernels"]
        )
    with col_top2:
        ad_method = st.selectbox("AD手法", ["knn", "ocsvm", "ocsvm_gamma_optimization"])
    with col_top3:
        fold_number = st.number_input("CV分割数 (KFold)", min_value=3, max_value=20, value=10, step=1)

    # 追加オプション
    with st.expander("高度な設定（必要なときだけ）", expanded=False):
        st.caption("SVR/GPRの設定や候補点生成の件数などを調整できます。")
        linear_svr_cs = 2 ** np.arange(-10, 5, dtype=float)
        linear_svr_eps = 2 ** np.arange(-10, 0, dtype=float)
        nonlinear_svr_cs = 2 ** np.arange(-5, 10, dtype=float)
        nonlinear_svr_eps = 2 ** np.arange(-10, 0, dtype=float)
        nonlinear_svr_gammas = 2 ** np.arange(-20, 10, dtype=float)
        ocsvm_nu = st.number_input("OCSVM ν", min_value=0.001, max_value=0.5, value=0.04, step=0.01)
        ocsvm_gamma_default = st.number_input("OCSVM γ (固定モード)", min_value=1e-6, max_value=10.0, value=0.1, step=0.1, format="%.6f")
        rate_inside_ad = st.slider("ADしきい値（kNNで内側に含める割合 / OCSVMは無視）", 0.5, 0.999, 0.96, 0.005)
        n_candidates = st.number_input("候補点のサンプル数（未評価）", min_value=50, max_value=5000, value=500, step=50)
        gpr_kernel_index = st.number_input("GPR単一カーネル番号 (0-10)", min_value=0, max_value=10, value=2, step=1)

    # データ点数チェック
    df_valid = df.dropna(subset=selected_conditions + ["結果"]).copy()
    if df_valid.empty or len(df_valid) < 3 or len(selected_conditions) == 0:
        st.warning("📉 有効なデータ点が不足しています（少なくとも3点、条件列1つ以上）。")
        st.stop()

    # 入出力データ
    X = df_valid[selected_conditions].values
    y = df_valid["結果"].values

    # 非線形変換（OLSのみ）
    x_tmp, x_pred_tmp = None, None
    if regression_method == "ols_nonlinear":
        x_tmp = pd.DataFrame(X, columns=selected_conditions)
        x_square = x_tmp ** 2
        X_aug = []
        col_names = []

        # 二乗 & 交差項
        for i in range(x_tmp.shape[1]):
            for j in range(x_tmp.shape[1]):
                if i == j:
                    X_aug.append(x_square.iloc[:, i].values)
                    col_names.append(f"{x_square.columns[i]}^2")
                elif i < j:
                    X_aug.append((x_tmp.iloc[:, i] * x_tmp.iloc[:, j]).values)
                    col_names.append(f"{x_tmp.columns[i]}*{x_tmp.columns[j]}")
        if X_aug:
            X = np.column_stack([X] + X_aug)
            selected_conditions = selected_conditions + col_names  # 拡張名を列名として継ぎ足す

    # 特徴の標準偏差ゼロを除去
    X_df = pd.DataFrame(X, columns=selected_conditions)
    deleting_cols = X_df.columns[X_df.std(axis=0, ddof=0) == 0.0].tolist()
    if len(deleting_cols) > 0:
        X_df = X_df.drop(columns=deleting_cols)
        selected_conditions = [c for c in selected_conditions if c not in deleting_cols]
    X = X_df.values

    # オートスケーリング
    X_auto, y_auto, scalers = autoscale_fit_transform(X, y)

    # --------------------------------------------------------
    # モデル構築
    # --------------------------------------------------------
    model = None
    kernels = build_kernels(X_auto.shape[1])
    if regression_method == "ols_linear" or regression_method == "ols_nonlinear":
        model = LinearRegression()

    elif regression_method == "svr_linear":
        # C, epsilon をCV最適化
        cv = KFold(n_splits=int(fold_number), shuffle=True, random_state=9)
        grid = {"C": linear_svr_cs, "epsilon": linear_svr_eps}
        gs = GridSearchCV(SVR(kernel="linear"), grid, cv=cv)
        gs.fit(X_auto, y_auto)
        model = SVR(kernel="linear", C=gs.best_params_["C"], epsilon=gs.best_params_["epsilon"])

    elif regression_method == "svr_gaussian":
        # γ は Gram 分散最大化 → その後 epsilon, C をCV選択
        optimal_gamma = gamma_by_gram_variance(X_auto, 2 ** np.arange(-20, 10, dtype=float))
        cv = KFold(n_splits=int(fold_number), shuffle=True, random_state=9)
        # epsilon 最適化
        r2_list = []
        for eps in 2 ** np.arange(-10, 0, dtype=float):
            m = SVR(kernel="rbf", C=3, epsilon=eps, gamma=optimal_gamma)
            ycv = cross_val_predict(m, X_auto, y_auto, cv=cv)
            r2_list.append(r2_score(y, inverse_scale_y(ycv, scalers)))
        optimal_eps = 2 ** np.arange(-10, 0, dtype=float)[int(np.argmax(r2_list))]
        # C 最適化
        r2_list = []
        for Cc in 2 ** np.arange(-5, 10, dtype=float):
            m = SVR(kernel="rbf", C=Cc, epsilon=optimal_eps, gamma=optimal_gamma)
            ycv = cross_val_predict(m, X_auto, y_auto, cv=cv)
            r2_list.append(r2_score(y, inverse_scale_y(ycv, scalers)))
        optimal_C = 2 ** np.arange(-5, 10, dtype=float)[int(np.argmax(r2_list))]
        # γ 再探索
        r2_list = []
        for gg in 2 ** np.arange(-20, 10, dtype=float):
            m = SVR(kernel="rbf", C=optimal_C, epsilon=optimal_eps, gamma=gg)
            ycv = cross_val_predict(m, X_auto, y_auto, cv=cv)
            r2_list.append(r2_score(y, inverse_scale_y(ycv, scalers)))
        optimal_gamma = 2 ** np.arange(-20, 10, dtype=float)[int(np.argmax(r2_list))]
        model = SVR(kernel="rbf", C=optimal_C, epsilon=optimal_eps, gamma=optimal_gamma)

    elif regression_method == "gpr_one_kernel":
        selected_kernel = kernels[int(gpr_kernel_index)]
        model = GaussianProcessRegressor(alpha=0, kernel=selected_kernel)

    elif regression_method == "gpr_kernels":
        cv = KFold(n_splits=int(fold_number), shuffle=True, random_state=9)
        r2cvs = []
        best_k_idx = 0
        for idx, k in enumerate(kernels):
            m = GaussianProcessRegressor(alpha=0, kernel=k)
            ycv = cross_val_predict(m, X_auto, y_auto, cv=cv)
            r2cvs.append(r2_score(y, inverse_scale_y(ycv, scalers)))
            if r2cvs[-1] >= np.max(r2cvs):
                best_k_idx = idx
        st.info(f"CVで選ばれたカーネル番号: {best_k_idx}")
        model = GaussianProcessRegressor(alpha=0, kernel=kernels[best_k_idx])

    # 学習
    model.fit(X_auto, y_auto)

    # --------------------------------------------------------
    # 学習性能 & CV 性能
    # --------------------------------------------------------
    y_pred_train = inverse_scale_y(model.predict(X_auto), scalers)
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("R² (train)", f"{r2_score(y, y_pred_train):.4f}")
    with col_m2:
        st.metric("RMSE (train)", f"{mean_squared_error(y, y_pred_train, squared=False):.6g}")
    with col_m3:
        st.metric("MAE (train)", f"{mean_absolute_error(y, y_pred_train):.6g}")

    cv = KFold(n_splits=int(fold_number), shuffle=True, random_state=9)
    y_cv_scaled = cross_val_predict(model, X_auto, y_auto, cv=cv)
    y_cv = inverse_scale_y(y_cv_scaled, scalers)

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.metric("R² (CV)", f"{r2_score(y, y_cv):.4f}")
    with col_c2:
        st.metric("RMSE (CV)", f"{mean_squared_error(y, y_cv, squared=False):.6g}")
    with col_c3:
        st.metric("MAE (CV)", f"{mean_absolute_error(y, y_cv):.6g}")

    # 実測 vs CV推定 プロット
    fig_cv, ax_cv = plt.subplots()
    ax_cv.scatter(y, y_cv, c="blue")
    y_max = max(np.max(y), np.max(y_cv))
    y_min = min(np.min(y), np.min(y_cv))
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax_cv.plot([y_min - pad, y_max + pad], [y_min - pad, y_max + pad], "k-")
    ax_cv.set_xlabel("実測値")
    ax_cv.set_ylabel("CV推定値")
    ax_cv.set_aspect("equal", adjustable="box")
    ax_cv.set_title("実測 vs CV推定")
    st.pyplot(fig_cv)

    # --------------------------------------------------------
    # 未評価候補 生成 & 予測（GPRは不確かさも）
    # --------------------------------------------------------
    st.markdown("---")
    st.subheader("🔮 次のサンプル提案（候補生成 → 予測 → ADでフィルタ）")

    # 候補生成（min-max一様）
    base_cond_cols = [c for c in df.columns if c.startswith("条件")]  # 元の物理条件だけ
    if len(base_cond_cols) == 0:
        st.warning("候補生成のための条件列が見つかりません。")
        st.stop()

    cand_df = make_candidates_uniform(df.dropna(subset=base_cond_cols + ["結果"]), base_cond_cols, int(n_candidates))
    # 解析に使った特徴と整合を取る（OLS_nonlinear では拡張特徴が必要）
    Xc = cand_df[[c for c in base_cond_cols if c in cand_df.columns]].copy()

    if regression_method == "ols_nonlinear":
        # 元の選択条件（拡張前）を推定
        # 交差項を再構成
        base_cols_used = [c for c in df_valid.columns if c.startswith("条件")]
        x_pred_tmp = Xc[base_cols_used].copy()
        x_square_pred = x_pred_tmp ** 2
        X_aug = []
        col_names = []

        for i in range(x_pred_tmp.shape[1]):
            for j in range(x_pred_tmp.shape[1]):
                if i == j:
                    X_aug.append(x_square_pred.iloc[:, i].values)
                    col_names.append(f"{x_square_pred.columns[i]}^2")
                elif i < j:
                    X_aug.append((x_pred_tmp.iloc[:, i] * x_pred_tmp.iloc[:, j]).values)
                    col_names.append(f"{x_pred_tmp.columns[i]}*{x_pred_tmp.columns[j]}")
        if X_aug:
            Xc_full = np.column_stack([x_pred_tmp.values] + X_aug)
            Xc_df = pd.DataFrame(Xc_full, columns=base_cols_used + col_names)
        else:
            Xc_df = x_pred_tmp.copy()
        # 学習時に落とした0分散列を合わせる
        for col in [c for c in selected_conditions if c not in Xc_df.columns]:
            # 欠けている拡張列がある場合は、0で埋める（実用上の近似）
            Xc_df[col] = 0.0
        Xc_df = Xc_df[selected_conditions]
    else:
        # 学習で残った列順に合わせる
        missing_cols = [c for c in selected_conditions if c not in Xc.columns]
        for c in missing_cols:
            # 拡張列は0で埋める（OLS_nonlinear 以外では基本発生しない）
            Xc[c] = 0.0
        Xc_df = Xc[selected_conditions]

    Xc_auto = autoscale_transform(Xc_df.values, scalers)

    if isinstance(model, GaussianProcessRegressor):
        y_pred_c_scaled, y_std_c_scaled = model.predict(Xc_auto, return_std=True)
        y_pred_c = inverse_scale_y(y_pred_c_scaled, scalers)
        y_std_c = y_std_c_scaled * scalers[3]  # y_std
    else:
        y_pred_c_scaled = model.predict(Xc_auto)
        y_pred_c = inverse_scale_y(y_pred_c_scaled, scalers)
        y_std_c = None

    # --------------------------------------------------------
    # AD 判定（候補にも適用）
    # --------------------------------------------------------
    st.markdown("#### 適用領域 (AD) 判定")
    inside_flag_pred = None
    ad_index_pred = None

    if ad_method == "knn":
        k = 5
        ad_model = NearestNeighbors(n_neighbors=k, metric="euclidean")
        ad_model.fit(X_auto)
        # 学習データの閾値決定
        dist_train, _ = ad_model.kneighbors(X_auto, n_neighbors=k+1)  # 自身を含むので +1
        mean_dist_train = dist_train[:, 1:].mean(axis=1)
        thr = np.sort(mean_dist_train)[int(round(len(mean_dist_train)*rate_inside_ad)) - 1]
        # 候補のAD
        dist_pred, _ = ad_model.kneighbors(Xc_auto, n_neighbors=k)
        mean_dist_pred = dist_pred.mean(axis=1)
        inside_flag_pred = (mean_dist_pred <= thr)
        ad_index_pred = mean_dist_pred  # 小さいほど内側

        st.caption("学習データの kNN 平均距離分布")
        safe_hist(mean_dist_train, "kNN 平均距離（学習）", "距離")
        st.caption("候補データの kNN 平均距離分布")
        safe_hist(mean_dist_pred, "kNN 平均距離（候補）", "距離")

    else:
        # OCSVM 系
        if ad_method == "ocsvm_gamma_optimization":
            gamma_grid = 2 ** np.arange(-20, 11, dtype=float)
            optimal_gamma = gamma_by_gram_variance(X_auto, gamma_grid)
            st.write(f"OCSVM 最適化された γ : {optimal_gamma:g}")
            gamma = optimal_gamma
        else:
            gamma = ocsvm_gamma_default

        ad_model = OneClassSVM(kernel="rbf", gamma=gamma, nu=ocsvm_nu)
        ad_model.fit(X_auto)

        # 学習データの密度（参考）
        dd_train = ad_model.decision_function(X_auto)
        st.caption("学習データの OCSVM decision_function 分布")
        safe_hist(dd_train, "OCSVM f(x)（学習）", "f(x)")

        # 候補
        dd_pred = ad_model.decision_function(Xc_auto)
        inside_flag_pred = (dd_pred >= 0)
        ad_index_pred = dd_pred  # 大きいほど内側

        st.caption("候補データの OCSVM decision_function 分布")
        safe_hist(dd_pred, "OCSVM f(x)（候補）", "f(x)")

    # --------------------------------------------------------
    # 目的に応じたスコアと上位提案
    # --------------------------------------------------------
    st.markdown("#### 提案結果")
    if mode == "最大化":
        score = y_pred_c.copy()
    else:  # 最小化
        score = -y_pred_c.copy()

    # AD外は極端に悪いスコアに
    score_filtered = score.copy()
    score_filtered[~inside_flag_pred] = -1e20

    # 上位案の抽出
    top_k = st.number_input("表示する上位提案数", 1, 50, 5, 1)
    idx_sorted = np.argsort(-score_filtered)  # 大きい順
    idx_sorted = [i for i in idx_sorted if inside_flag_pred[i]][:top_k]

    if len(idx_sorted) == 0:
        st.error("AD内の候補がありませんでした。サンプル数を増やすか、AD設定を緩めてください。")
    else:
        # テーブル整備
        out_rows = []
        for i in idx_sorted:
            row = {col: cand_df.iloc[i][col] if col in cand_df.columns else np.nan for col in cand_df.columns}
            row["予測値"] = y_pred_c[i]
            if y_std_c is not None:
                row["予測std"] = y_std_c[i]
            row["AD_index"] = ad_index_pred[i]
            out_rows.append(row)
        out_df = pd.DataFrame(out_rows)
        st.dataframe(out_df, use_container_width=True)

        # 先頭を「推奨 #1」として表示
        best_idx = idx_sorted[0]
        best_row = cand_df.iloc[best_idx]
        st.success("🚀 推奨 #1（AD内）: " + ", ".join([f"{c}={best_row[c]:.4f}" for c in cand_df.columns]))
        st.write(f"予測値: {y_pred_c[best_idx]:.6g}" + (f", 予測std: {y_std_c[best_idx]:.6g}" if y_std_c is not None else ""))

        # ダウンロード用
        st.download_button(
            "⬇️ 上位提案（CSV）をダウンロード",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="proposals_top.csv",
            mime="text/csv",
        )

    # --------------------------------------------------------
    # 可視化（1次元/履歴）
    # --------------------------------------------------------
    st.markdown("---")
    viz_mode = st.radio("可視化", ["散布図（1軸）", "最適化履歴（CV推定値）"], horizontal=True)

    if viz_mode == "散布図（1軸）":
        if len([c for c in df_valid.columns if c.startswith("条件")]) == 0:
            st.info("散布図のための条件列がありません。")
        else:
            xcol = st.selectbox("横軸にする条件列", [c for c in df_valid.columns if c.startswith("条件")], index=0)
            fig, ax = plt.subplots()
            ax.scatter(df_valid[xcol], df_valid["結果"], c="blue", label="実験データ")
            ax.set_xlabel(xcol)
            ax.set_ylabel("結果")
            ax.set_title(f"{xcol} vs 結果")
            st.pyplot(fig)

    else:  # 履歴（CV推定値の推移を単に並べる）
        fig, ax = plt.subplots()
        ax.plot(range(1, len(y_cv) + 1), y_cv, marker="o")
        ax.set_xlabel("データ番号")
        ax.set_ylabel("CV推定値")
        ax.set_title("CV推定値の履歴")
        st.pyplot(fig)
