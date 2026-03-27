import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib, warnings
warnings.filterwarnings("ignore")
from data_loader import load_data, get_feature_matrix

@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def train_regressors():
    df = get_data()
    X, feat_cols = get_feature_matrix(df)
    y = np.log1p(df["estimated_annual_spend"].values)   # log-transform for skew

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Random Forest Reg": RandomForestRegressor(n_estimators=200, max_depth=8,
                                                    random_state=42, n_jobs=-1),
    }

    results, trained = {}, {}
    for name, mdl in models.items():
        if name == "Random Forest Reg":
            mdl.fit(X_tr, y_tr)
            y_pred = mdl.predict(X_te)
        else:
            mdl.fit(X_tr_sc, y_tr)
            y_pred = mdl.predict(X_te_sc)

        y_pred_rs = np.expm1(y_pred)
        y_te_rs   = np.expm1(y_te)
        results[name] = {
            "rmse":    np.sqrt(mean_squared_error(y_te_rs, y_pred_rs)),
            "mae":     mean_absolute_error(y_te_rs, y_pred_rs),
            "r2":      r2_score(y_te, y_pred),
            "y_pred":  y_pred_rs,
            "y_actual":y_te_rs,
        }
        trained[name] = (mdl, scaler if name != "Random Forest Reg" else None)

    # Save best model
    best_mdl, best_sc = trained["Random Forest Reg"]
    joblib.dump({"model": best_mdl, "scaler": None, "feat_cols": feat_cols,
                 "log_transform": True}, "zaria_reg_model.pkl")

    return trained, results, feat_cols, X, df

def render():
    st.title("💰 Regression & Customer Lifetime Value")
    st.caption("How much will a customer spend annually? — Linear · Ridge · Random Forest Regressor")

    with st.spinner("Training regression models…"):
        trained, results, feat_cols, X_full, df = train_regressors()

    # ── Model Comparison ─────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Model Performance Comparison</div>", unsafe_allow_html=True)

    metrics_df = pd.DataFrame({
        "Model": list(results.keys()),
        "RMSE (₹)": [f"₹{results[m]['rmse']:,.0f}" for m in results],
        "MAE (₹)":  [f"₹{results[m]['mae']:,.0f}"  for m in results],
        "R² Score": [f"{results[m]['r2']:.4f}"      for m in results],
    })
    st.dataframe(metrics_df, use_container_width=True)

    fig_cmp = px.bar(
        pd.DataFrame({
            "Model":    list(results.keys()),
            "R² Score": [results[m]["r2"] for m in results],
        }),
        x="Model", y="R² Score", color="Model",
        color_discrete_sequence=["#1D9E75","#378ADD","#BA7517"],
        text_auto=".3f", title="R² Score Comparison (higher = better fit)")
    fig_cmp.update_traces(textposition="outside")
    fig_cmp.update_layout(height=320, showlegend=False,
                          margin=dict(t=50,b=10,l=10,r=10), yaxis_range=[0,1])
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Model deep dive ───────────────────────────────────────────────────────
    sel = st.selectbox("Select model for detailed analysis", list(results.keys()))
    res = results[sel]

    c1,c2,c3 = st.columns(3)
    c1.metric("RMSE", f"₹{res['rmse']:,.0f}")
    c2.metric("MAE",  f"₹{res['mae']:,.0f}")
    c3.metric("R²",   f"{res['r2']:.4f}")

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Actual vs Predicted Annual Spend</div>", unsafe_allow_html=True)
    ap_df = pd.DataFrame({"Actual (₹)": res["y_actual"], "Predicted (₹)": res["y_pred"]})
    # cap for readability
    ap_df = ap_df[ap_df["Actual (₹)"] < ap_df["Actual (₹)"].quantile(0.97)]
    fig_ap = px.scatter(ap_df, x="Actual (₹)", y="Predicted (₹)",
                        opacity=0.4, color_discrete_sequence=["#1D9E75"],
                        title=f"Actual vs Predicted — {sel}")
    max_val = max(ap_df["Actual (₹)"].max(), ap_df["Predicted (₹)"].max())
    fig_ap.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                mode="lines", line=dict(dash="dash", color="red"),
                                name="Perfect fit"))
    fig_ap.update_layout(height=420, margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig_ap, use_container_width=True)

    # ── Residual Plot ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Residual Analysis</div>", unsafe_allow_html=True)
    ap_df["Residual (₹)"] = ap_df["Predicted (₹)"] - ap_df["Actual (₹)"]
    c_r1, c_r2 = st.columns(2)
    with c_r1:
        fig_res = px.scatter(ap_df, x="Predicted (₹)", y="Residual (₹)",
                             opacity=0.4, color_discrete_sequence=["#378ADD"],
                             title="Residuals vs Predicted")
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        fig_res.update_layout(height=340, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_res, use_container_width=True)

    with c_r2:
        fig_rh = px.histogram(ap_df, x="Residual (₹)", nbins=30,
                              color_discrete_sequence=["#BA7517"],
                              title="Residual Distribution")
        fig_rh.update_layout(height=340, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_rh, use_container_width=True)

    # ── Feature Importance (RF) ───────────────────────────────────────────────
    rf_model, _ = trained["Random Forest Reg"]
    st.markdown("<div class='section-hdr'>Feature Importance — Random Forest Regressor</div>", unsafe_allow_html=True)
    clean = [c.replace("_enc","").replace("_"," ").title() for c in feat_cols]
    fi_df = pd.DataFrame({"Feature": clean, "Importance": rf_model.feature_importances_})
    fi_df = fi_df.sort_values("Importance", ascending=True).tail(15)
    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale=["#FAC775","#412402"],
                    text="Importance", title="Top 15 Features for Spend Prediction")
    fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_fi.update_layout(height=460, coloraxis_showscale=False,
                         margin=dict(t=50,b=10,l=10,r=10), yaxis_title="")
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── CLV Ranking & Tiering ─────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Customer Lifetime Value — Tiering & Playbook</div>", unsafe_allow_html=True)

    df_full = df.copy()
    from data_loader import get_feature_matrix as gfm
    X_all, _ = gfm(df_full)
    rf_full, _ = trained["Random Forest Reg"]
    df_full["predicted_annual_spend"] = np.expm1(rf_full.predict(X_all)).round(-1)

    p80 = df_full["predicted_annual_spend"].quantile(0.80)
    p30 = df_full["predicted_annual_spend"].quantile(0.30)
    df_full["clv_tier"] = pd.cut(df_full["predicted_annual_spend"],
                                  bins=[0, p30, p80, 1e9],
                                  labels=["Standard","Mid-Value","VIP"])

    tier_counts = df_full["clv_tier"].value_counts().reset_index()
    tier_counts.columns = ["Tier","Count"]
    c_t1, c_t2 = st.columns(2)
    with c_t1:
        fig_tc = px.pie(tier_counts, names="Tier", values="Count",
                        color_discrete_sequence=["#888780","#378ADD","#1D9E75"],
                        title="CLV Tier Distribution", hole=0.4)
        fig_tc.update_layout(height=320, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_tc, use_container_width=True)

    with c_t2:
        spend_by_tier = df_full.groupby("clv_tier")["predicted_annual_spend"].mean().round(0).reset_index()
        spend_by_tier.columns = ["Tier","Avg Predicted Spend (₹)"]
        fig_st = px.bar(spend_by_tier, x="Tier", y="Avg Predicted Spend (₹)",
                        color="Tier",
                        color_discrete_sequence=["#888780","#378ADD","#1D9E75"],
                        text="Avg Predicted Spend (₹)",
                        title="Average Predicted Spend by Tier")
        fig_st.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        fig_st.update_layout(height=320, showlegend=False,
                             margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_st, use_container_width=True)

    # Playbook
    st.markdown("**Marketing playbook by CLV tier:**")
    col_v, col_m, col_s = st.columns(3)
    with col_v:
        st.markdown(f"""<div class='insight-box'>
        <b>🏆 VIP (top 20%)</b><br>
        Predicted spend > ₹{p80:,.0f}<br>
        Count: {(df_full['clv_tier']=='VIP').sum()}<br><br>
        ✅ Free express delivery<br>
        ✅ Personal stylist WhatsApp<br>
        ✅ Early collection preview<br>
        ✅ Handwritten thank-you note
        </div>""", unsafe_allow_html=True)
    with col_m:
        st.markdown(f"""<div class='warning-box'>
        <b>🥈 Mid-Value (middle 50%)</b><br>
        Spend ₹{p30:,.0f} – ₹{p80:,.0f}<br>
        Count: {(df_full['clv_tier']=='Mid-Value').sum()}<br><br>
        ✅ Email campaigns<br>
        ✅ Instagram content<br>
        ✅ Seasonal discount offers<br>
        ✅ Loyalty points programme
        </div>""", unsafe_allow_html=True)
    with col_s:
        st.markdown("""<div class='metric-card' style='text-align:left;padding:12px'>
        <b>📢 Standard (bottom 30%)</b><br><br>
        ✅ WhatsApp broadcast only<br>
        ✅ Festival sale announcements<br>
        ✅ No paid retargeting<br>
        ✅ Re-evaluate after 6 months
        </div>""", unsafe_allow_html=True)

    # Top 20 CLV customers
    with st.expander("🏆 View Top 20 Highest CLV Customers"):
        show = ["age_group","region","city_tier","fashion_identity",
                "brand_openness","monthly_income_band","predicted_annual_spend","clv_tier"]
        show = [c for c in show if c in df_full.columns]
        st.dataframe(df_full.sort_values("predicted_annual_spend", ascending=False)[show].head(20).reset_index(drop=True),
                     use_container_width=True)

    st.download_button("Download CLV Rankings",
                       df_full[show + ["predicted_annual_spend"]].to_csv(index=False).encode(),
                       "zaria_clv_rankings.csv","text/csv")
