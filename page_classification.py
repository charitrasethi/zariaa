import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc,
                              classification_report)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings, joblib, os
warnings.filterwarnings("ignore")
from data_loader import load_data, get_feature_matrix

@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def train_models():
    df = get_data()
    X, feat_cols = get_feature_matrix(df)
    y_raw = df["zaria_interest_label"].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8,
                                                 random_state=42, n_jobs=-1),
        "XGBoost":       XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                                       random_state=42, eval_metric="mlogloss",
                                       use_label_encoder=False, verbosity=0),
        "Logistic Reg":  LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    }
    results, trained = {}, {}
    for name, model in models.items():
        model.fit(X_tr_sm, y_tr_sm)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)
        results[name] = {
            "accuracy":  accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, average="weighted", zero_division=0),
            "recall":    recall_score(y_te, y_pred, average="weighted", zero_division=0),
            "f1":        f1_score(y_te, y_pred, average="weighted", zero_division=0),
            "y_pred":    y_pred,
            "y_prob":    y_prob,
            "cm":        confusion_matrix(y_te, y_pred),
        }
        trained[name] = model

    # Save best model (RF) for predictor page
    best = trained["Random Forest"]
    joblib.dump({"model": best, "le": le, "feat_cols": feat_cols}, "zaria_clf_model.pkl")

    return trained, results, X_te, y_te, le, feat_cols, X

def render():
    st.title("🎯 Classification")
    st.caption("Will a customer buy from Zaria? — Random Forest · XGBoost · Logistic Regression")

    with st.spinner("Training classifiers (SMOTE applied for class balance)…"):
        trained, results, X_te, y_te, le, feat_cols, X_full = train_models()

    classes = le.classes_

    # ── Model Comparison ─────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Model Performance Comparison</div>", unsafe_allow_html=True)

    metrics_df = pd.DataFrame({
        "Model":     list(results.keys()),
        "Accuracy":  [results[m]["accuracy"]  for m in results],
        "Precision": [results[m]["precision"] for m in results],
        "Recall":    [results[m]["recall"]    for m in results],
        "F1-Score":  [results[m]["f1"]        for m in results],
    }).round(4)

    # Highlight best per metric
    def highlight_max(s):
        return ["background-color:#d4edda; font-weight:bold"
                if v == s.max() else "" for v in s]

    st.dataframe(metrics_df.style.apply(highlight_max, subset=["Accuracy","Precision","Recall","F1-Score"]),
                 use_container_width=True)

    # Bar chart comparison
    metrics_long = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig_cmp = px.bar(metrics_long, x="Metric", y="Score", color="Model",
                     barmode="group", text_auto=".3f",
                     color_discrete_sequence=["#1D9E75","#378ADD","#BA7517"],
                     title="Accuracy · Precision · Recall · F1-Score by Model")
    fig_cmp.update_traces(textposition="outside")
    fig_cmp.update_layout(height=380, yaxis_range=[0,1.1],
                          margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Select model for deep-dive ────────────────────────────────────────────
    sel_model = st.selectbox("Select model for detailed analysis", list(results.keys()))
    res = results[sel_model]
    model_obj = trained[sel_model]

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{res['accuracy']:.3f}")
    col2.metric("Precision", f"{res['precision']:.3f}")
    col3.metric("Recall",    f"{res['recall']:.3f}")
    col4.metric("F1-Score",  f"{res['f1']:.3f}")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Confusion Matrix</div>", unsafe_allow_html=True)
    c_cm, c_roc = st.columns(2)
    with c_cm:
        cm = res["cm"]
        fig_cm, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    linewidths=0.5, annot_kws={"size":12})
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title(f"Confusion Matrix — {sel_model}", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)
        plt.close()

    # ── ROC Curves ───────────────────────────────────────────────────────────
    with c_roc:
        st.markdown("**ROC Curves (one-vs-rest)**")
        y_te_bin = label_binarize(y_te, classes=list(range(len(classes))))
        y_prob   = res["y_prob"]

        fig_roc = go.Figure()
        colors  = ["#1D9E75","#378ADD","#E24B4A"]
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_te_bin[:,i], y_prob[:,i])
            roc_auc = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"{cls} (AUC={roc_auc:.2f})",
                                         line=dict(color=colors[i], width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(dash="dash",color="gray"), name="Random"))
        fig_roc.update_layout(title=f"ROC Curves — {sel_model}",
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate",
                              height=380, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Feature Importance</div>", unsafe_allow_html=True)

    if hasattr(model_obj, "feature_importances_"):
        importances = model_obj.feature_importances_
    elif hasattr(model_obj, "coef_"):
        importances = np.abs(model_obj.coef_).mean(axis=0)
    else:
        importances = np.zeros(len(feat_cols))

    clean_names = [c.replace("_enc","").replace("_"," ").title() for c in feat_cols]
    fi_df = pd.DataFrame({"Feature": clean_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=True).tail(20)

    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale=["#9FE1CB","#085041"],
                    text="Importance",
                    title=f"Top 20 Feature Importances — {sel_model}")
    fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_fi.update_layout(height=520, coloraxis_showscale=False,
                         margin=dict(t=50,b=10,l=10,r=10), yaxis_title="")
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("<div class='insight-box'>🔑 The top features reveal EXACTLY what drives purchase intent for Zaria. Invest in messaging and product decisions around these signals.</div>", unsafe_allow_html=True)

    # ── Classification Report ─────────────────────────────────────────────────
    with st.expander("📋 Full Classification Report"):
        cr = classification_report(y_te, res["y_pred"], target_names=classes, output_dict=True)
        cr_df = pd.DataFrame(cr).T.round(3)
        st.dataframe(cr_df, use_container_width=True)

    # ── Hot Leads Table ───────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Top 50 High-Probability Leads</div>", unsafe_allow_html=True)
    df_full = get_data()
    X_all, _ = get_feature_matrix(df_full)
    probs = trained["Random Forest"].predict_proba(X_all)
    int_idx = list(classes).index("Interested") if "Interested" in classes else 0
    df_full["interest_probability"] = probs[:, int_idx].round(3)
    df_full["predicted_label"] = le.inverse_transform(trained["Random Forest"].predict(X_all))

    leads = df_full.sort_values("interest_probability", ascending=False).head(50)
    show_cols = ["age_group","region","city_tier","fashion_identity","brand_openness",
                 "conversion_trigger","monthly_income_band","interest_probability","predicted_label"]
    show_cols = [c for c in show_cols if c in leads.columns]
    st.dataframe(leads[show_cols].reset_index(drop=True), use_container_width=True)

    st.download_button("Download All Predictions",
                       df_full[show_cols].to_csv(index=False).encode(),
                       "zaria_predictions.csv","text/csv")

    st.markdown("<div class='warning-box'>⚙️ Model saved to <code>zaria_clf_model.pkl</code> — used by the New Customer Predictor page.</div>", unsafe_allow_html=True)
