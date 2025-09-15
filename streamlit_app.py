with tab2:
    st.subheader("✨ 次の実験条件を提案します")

    df = pd.read_csv(CSV_FILE)
    all_condition_cols = [col for col in df.columns if col.startswith("条件")]

    if not all_condition_cols:
        st.warning("⚠️ 条件データが存在しません。")
    else:
        selected_conditions = st.multiselect(
            "解析に使用する条件を選択",
            all_condition_cols,
            default=all_condition_cols[:1]
        )

        regression_method = st.selectbox("回帰手法を選択", 
                                         ["ols_linear", "ols_nonlinear", 
                                          "svr_linear", "svr_gaussian", 
                                          "gpr_one_kernel"])
        ad_method = st.selectbox("AD手法を選択", ["knn", "ocsvm", "ocsvm_gamma_optimization"])

        if df.empty or len(df) < 3:
            st.warning("📉 データ点数が少ないです。もう少しデータを追加してください。")
        else:
            if st.button("🚀 解析スタート"):
                df_valid = df.dropna(subset=selected_conditions + ["結果"])
                X = df_valid[selected_conditions].values
                y = df_valid["結果"].values

                # ---- Kaneko式 回帰モデル構築 ----
                from sklearn.linear_model import LinearRegression
                from sklearn.svm import SVR, OneClassSVM
                from sklearn.model_selection import KFold, cross_val_predict
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel
                from sklearn.neighbors import NearestNeighbors
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

                # autoscaling
                autoscaled_X = (X - X.mean(axis=0)) / X.std(axis=0)
                autoscaled_y = (y - y.mean()) / y.std()

                # モデル選択
                if regression_method == "ols_linear":
                    model = LinearRegression()
                elif regression_method == "svr_linear":
                    model = SVR(kernel="linear")
                elif regression_method == "svr_gaussian":
                    model = SVR(kernel="rbf")
                elif regression_method == "gpr_one_kernel":
                    kernel = ConstantKernel()*RBF() + WhiteKernel()
                    model = GaussianProcessRegressor(kernel=kernel, alpha=0)

                model.fit(autoscaled_X, autoscaled_y)

                # 予測
                y_pred = model.predict(autoscaled_X)*y.std() + y.mean()

                # 評価指標
                st.write("**学習データでの評価**")
                st.write("R²:", r2_score(y, y_pred))
                st.write("RMSE:", mean_squared_error(y, y_pred, squared=False))
                st.write("MAE:", mean_absolute_error(y, y_pred))

                # CV
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                y_cv = cross_val_predict(model, autoscaled_X, autoscaled_y, cv=cv)
                y_cv = y_cv*y.std() + y.mean()
                st.write("**CVでの評価**")
                st.write("R²:", r2_score(y, y_cv))
                st.write("RMSE:", mean_squared_error(y, y_cv, squared=False))

                # プロット
                fig, ax = plt.subplots()
                ax.scatter(y, y_cv, c="blue")
                ax.set_xlabel("実測値")
                ax.set_ylabel("CV推定値")
                st.pyplot(fig)

                # ---- AD判定 ----
                st.write("**適用領域 (AD) 判定**")
                if ad_method == "knn":
                    ad_model = NearestNeighbors(n_neighbors=3)
                    ad_model.fit(autoscaled_X)
                    dist, _ = ad_model.kneighbors(autoscaled_X)
                    mean_dist = dist[:,1:].mean(axis=1)
                    st.histogram_chart(pd.DataFrame(mean_dist, columns=["距離"]))
                elif ad_method == "ocsvm":
                    ad_model = OneClassSVM(kernel="rbf", gamma=0.1, nu=0.04)
                    ad_model.fit(autoscaled_X)
                    flag = ad_model.predict(autoscaled_X)
                    st.bar_chart(pd.Series(flag).value_counts())

                # ---- 次のサンプル提案 ----
                st.success("🔮 次のサンプル候補は、予測値が最大となる条件です。")
                idx_next = np.argmax(y_pred)
                st.write({col: df_valid.iloc[idx_next][col] for col in selected_conditions})
