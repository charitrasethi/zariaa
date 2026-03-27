import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, os, warnings
warnings.filterwarnings("ignore")
from data_loader import (load_data, get_feature_matrix, encode_features,
                         validate_upload, engineer_target, engineer_spend,
                         EXPECTED_COLS, BINARY_COLS)

CLUSTER_NAMES_MAP = {
    0:"Festival Splurger", 1:"Urban Fusion Millennial", 2:"Daily Cotton Homemaker",
    3:"Premium Occasion Shopper", 4:"Budget-Conscious Student", 5:"The Gifter",
}
CLUSTER_OFFERS = {
    "Festival Splurger":        ("Festival_Sale 28% off salwar suits","WhatsApp + local retail"),
    "Urban Fusion Millennial":  ("Buy2Get1 Indo-western + Palazzo bundle","Instagram Reels + D2C"),
    "Daily Cotton Homemaker":   ("Free shipping on cotton kurtis > ₹799","WhatsApp + word of mouth"),
    "Premium Occasion Shopper": ("Personalised offer + early collection access","Email + Brand website"),
    "Budget-Conscious Student": ("First order ₹150 off + COD available","Instagram + social commerce"),
    "The Gifter":               ("Gift bundle: kurti + bedding + gift wrap","WhatsApp + marketplace"),
}

@st.cache_data
def get_base_data():
    return load_data()

def load_clf():
    if os.path.exists("zaria_clf_model.pkl"):
        return joblib.load("zaria_clf_model.pkl")
    return None

def load_reg():
    if os.path.exists("zaria_reg_model.pkl"):
        return joblib.load("zaria_reg_model.pkl")
    return None

def assign_cluster(df_new):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    df_base = get_base_data()
    df_enc_base = encode_features(df_base)
    cluster_feat = ["fashion_identity_enc","price_sensitivity_enc","brand_openness_enc",
                    "online_purchase_confidence_enc","sustainability_consciousness_enc",
                    "purchase_frequency_enc"] + BINARY_COLS
    cluster_feat = [c for c in cluster_feat if c in df_enc_base.columns]
    X_base = df_enc_base[cluster_feat].fillna(0).values
    sc = StandardScaler()
    X_base_sc = sc.fit_transform(X_base)
    km = KMeans(n_clusters=6, random_state=42, n_init=10)
    km.fit(X_base_sc)

    df_enc_new = encode_features(df_new)
    cluster_feat_new = [c for c in cluster_feat if c in df_enc_new.columns]
    X_new = df_enc_new[cluster_feat_new].fillna(0).values
    if X_new.shape[1] < len(cluster_feat):
        X_new = np.hstack([X_new, np.zeros((len(X_new), len(cluster_feat)-X_new.shape[1]))])
    X_new_sc = sc.transform(X_new)
    labels = km.predict(X_new_sc)
    return [CLUSTER_NAMES_MAP.get(l, f"Cluster {l}") for l in labels]

def render():
    st.title("🆕 New Customer Predictor")
    st.caption("Upload new survey responses → get instant predictions, cluster assignments & personalised offers.")

    clf_loaded = load_clf()
    reg_loaded = load_reg()

    if clf_loaded is None or reg_loaded is None:
        st.warning("⚠️ Trained models not found. Please visit the **Classification** and **Regression & CLV** pages first to train and save models.")
        st.info("Once you visit those pages, models are auto-saved and this page will work.")

    st.markdown("---")

    # ── Template download ────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Step 1 — Download Template</div>", unsafe_allow_html=True)
    template_row = {
        "age_group":"25-34","region":"North_India","city_tier":"Metro",
        "occupation":"Salaried_Private","fashion_identity":"Fusion_Lover",
        "price_sensitivity":"Balanced","brand_openness":"Multi_Brand_Shopper",
        "online_purchase_confidence":"Fairly_Confident",
        "sustainability_consciousness":"Somewhat_Conscious",
        "purchase_frequency":"Every_2_3_Months","preferred_shopping_channel":"Online_Marketplace",
        "discovery_channel":"Instagram_Reels","conversion_trigger":"Customer_Reviews",
        "discount_preference":"Buy2_Get1_Bundle",
        "owns_kurti":1,"owns_salwar_suit":1,"owns_palazzo":0,"owns_indo_western":1,
        "owns_night_suit":0,"owns_bedding_set":0,"owns_saree":0,"owns_lehenga":0,
        "fabric_preference":"Pure_Cotton","color_preference":"Pastels",
        "monthly_income_band":"40K_70K",
    }
    template_df = pd.DataFrame([template_row])
    st.download_button("📥 Download CSV Template (25 columns)",
                       template_df.to_csv(index=False).encode(),
                       "zaria_upload_template.csv","text/csv")
    st.caption("Fill this template with your new survey responses and upload below.")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Step 2 — Upload Your Data</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV (25 survey columns)", type=["csv"])

    if uploaded is not None:
        try:
            df_new = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        # Validate
        st.markdown("<div class='section-hdr'>Step 3 — Validation Report</div>", unsafe_allow_html=True)
        errors, warnings_list = validate_upload(df_new)

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
            st.stop()
        else:
            st.success(f"✅ Validation passed — {len(df_new)} rows, {len(df_new.columns)} columns")
        for w in warnings_list:
            st.warning(f"⚠️ {w}")

        # Show column summary
        with st.expander("Preview uploaded data"):
            st.dataframe(df_new.head(10), use_container_width=True)

        # ── Predict ───────────────────────────────────────────────────────────
        st.markdown("<div class='section-hdr'>Step 4 — Running Predictions</div>", unsafe_allow_html=True)

        with st.spinner("Engineering features and running models…"):
            df_proc = engineer_target(df_new.copy())
            df_proc = engineer_spend(df_proc)

            # Classification
            results_df = df_new.copy()

            if clf_loaded:
                feat_cols_clf = clf_loaded["feat_cols"]
                df_enc = encode_features(df_new)
                X_clf = df_enc[[c for c in feat_cols_clf if c in df_enc.columns]].fillna(0)
                missing_clf = [c for c in feat_cols_clf if c not in df_enc.columns]
                for mc in missing_clf:
                    X_clf[mc] = 0
                X_clf = X_clf[feat_cols_clf].fillna(0)

                clf_model = clf_loaded["model"]
                le        = clf_loaded["le"]
                probs     = clf_model.predict_proba(X_clf)
                preds     = clf_model.predict(X_clf)
                int_idx   = list(le.classes_).index("Interested") if "Interested" in le.classes_ else 0

                results_df["interest_probability"] = probs[:, int_idx].round(3)
                results_df["predicted_interest"]   = le.inverse_transform(preds)
                results_df["lead_priority"] = pd.cut(
                    results_df["interest_probability"],
                    bins=[0, 0.40, 0.70, 1.01],
                    labels=["Low Priority","Nurture","Hot Lead"])
            else:
                results_df["interest_probability"] = 0.5
                results_df["predicted_interest"]   = "Unknown"
                results_df["lead_priority"]         = "Unknown"

            # Regression
            if reg_loaded:
                feat_cols_reg = reg_loaded["feat_cols"]
                df_enc2 = encode_features(df_new)
                X_reg = df_enc2[[c for c in feat_cols_reg if c in df_enc2.columns]].fillna(0)
                for mc2 in [c for c in feat_cols_reg if c not in df_enc2.columns]:
                    X_reg[mc2] = 0
                X_reg = X_reg[feat_cols_reg].fillna(0)
                reg_model = reg_loaded["model"]
                results_df["predicted_annual_spend"] = np.expm1(reg_model.predict(X_reg)).round(-1)
            else:
                results_df["predicted_annual_spend"] = df_proc["estimated_annual_spend"]

            # Clustering
            cluster_names = assign_cluster(df_new)
            results_df["cluster_assigned"] = cluster_names

            # Recommended offer + channel
            results_df["recommended_offer"]   = results_df["cluster_assigned"].map(
                lambda x: CLUSTER_OFFERS.get(x, ("Custom offer","—"))[0])
            results_df["recommended_channel"] = results_df["cluster_assigned"].map(
                lambda x: CLUSTER_OFFERS.get(x, ("—","Mixed"))[1])

            # Key trigger from conversion_trigger col
            results_df["key_trigger"] = df_new["conversion_trigger"] if "conversion_trigger" in df_new.columns else "—"

        # ── Summary KPIs ──────────────────────────────────────────────────────
        st.markdown("<div class='section-hdr'>Prediction Summary</div>", unsafe_allow_html=True)
        hot   = (results_df["lead_priority"] == "Hot Lead").sum()
        nurt  = (results_df["lead_priority"] == "Nurture").sum()
        low   = (results_df["lead_priority"] == "Low Priority").sum()
        avg_s = results_df["predicted_annual_spend"].mean()

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🔥 Hot Leads",     hot)
        c2.metric("🌱 Nurture",       nurt)
        c3.metric("📉 Low Priority",  low)
        c4.metric("💰 Avg Pred Spend",f"₹{avg_s:,.0f}")

        # Lead priority chart
        pri_df = results_df["lead_priority"].value_counts().reset_index()
        pri_df.columns = ["Priority","Count"]
        fig_pri = px.pie(pri_df, names="Priority", values="Count",
                         color_discrete_sequence=["#1D9E75","#FFC107","#E24B4A"],
                         title="Lead Priority Distribution", hole=0.4)
        fig_pri.update_layout(height=300, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_pri, use_container_width=True)

        # Cluster distribution
        cl_df = results_df["cluster_assigned"].value_counts().reset_index()
        cl_df.columns = ["Cluster","Count"]
        fig_cl = px.bar(cl_df, x="Cluster", y="Count", color="Cluster",
                        color_discrete_sequence=["#1D9E75","#378ADD","#BA7517",
                                                  "#534AB7","#E24B4A","#888780"],
                        text="Count", title="Cluster Assignment for New Customers")
        fig_cl.update_traces(textposition="outside")
        fig_cl.update_layout(height=320, showlegend=False,
                             margin=dict(t=50,b=10,l=10,r=10), xaxis_title="")
        st.plotly_chart(fig_cl, use_container_width=True)

        # ── Per-customer action cards ─────────────────────────────────────────
        st.markdown("<div class='section-hdr'>Per-Customer Action Cards</div>", unsafe_allow_html=True)
        display_cols = ["interest_probability","predicted_interest","lead_priority",
                        "cluster_assigned","predicted_annual_spend",
                        "recommended_offer","recommended_channel","key_trigger"]
        display_cols = [c for c in display_cols if c in results_df.columns]

        # Show top 10 hot leads as cards
        hot_leads = results_df[results_df["lead_priority"]=="Hot Lead"].head(5)
        if not hot_leads.empty:
            st.markdown("**🔥 Hot Leads — Act Now:**")
            for idx, row in hot_leads.iterrows():
                with st.expander(f"Customer #{idx+1} — Probability: {row.get('interest_probability',0):.0%} | Cluster: {row.get('cluster_assigned','—')}"):
                    col_a, col_b = st.columns(2)
                    col_a.metric("Interest Probability", f"{row.get('interest_probability',0):.0%}")
                    col_b.metric("Predicted Annual Spend", f"₹{row.get('predicted_annual_spend',0):,.0f}")
                    st.markdown(f"**Cluster:** {row.get('cluster_assigned','—')}")
                    st.markdown(f"**Recommended Offer:** {row.get('recommended_offer','—')}")
                    st.markdown(f"**Best Channel:** {row.get('recommended_channel','—')}")
                    st.markdown(f"**Conversion Trigger:** {row.get('key_trigger','—')}")

        # Full results table
        st.markdown("<div class='section-hdr'>Full Results Table</div>", unsafe_allow_html=True)
        st.dataframe(results_df[display_cols].reset_index(drop=True), use_container_width=True)

        # ── Download enriched CSV ─────────────────────────────────────────────
        all_output_cols = list(df_new.columns) + display_cols
        all_output_cols = list(dict.fromkeys(all_output_cols))
        output_df = results_df[all_output_cols] if all([c in results_df.columns for c in all_output_cols]) else results_df

        st.download_button(
            "📥 Download Enriched Predictions CSV",
            output_df.to_csv(index=False).encode(),
            "zaria_new_customer_predictions.csv","text/csv")

        st.markdown("<div class='insight-box'>✅ This file contains your original 25 survey columns PLUS: interest probability, lead priority, cluster assignment, predicted annual spend, recommended offer, channel, and conversion trigger. Import this into your CRM or WhatsApp broadcast tool.</div>", unsafe_allow_html=True)

    else:
        # Demo mode — show what the output looks like
        st.markdown("---")
        st.info("👆 Upload a CSV file to get started. Download the template above to see the required format.")
        st.markdown("**What you'll get for each new customer:**")
        demo_data = {
            "interest_probability": [0.87, 0.43, 0.21],
            "predicted_interest":   ["Interested","Neutral","Not_Interested"],
            "lead_priority":        ["Hot Lead","Nurture","Low Priority"],
            "cluster_assigned":     ["Urban Fusion Millennial","Festival Splurger","Budget-Conscious Student"],
            "predicted_annual_spend":[34500, 12000, 4200],
            "recommended_offer":    ["Buy2Get1 Indo-western bundle","Festival Sale 28% off","₹150 off first order"],
            "recommended_channel":  ["Instagram Reels","WhatsApp","Social Commerce"],
        }
        st.dataframe(pd.DataFrame(demo_data), use_container_width=True)
