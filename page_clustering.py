import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
from data_loader import load_data, encode_features, BINARY_COLS

@st.cache_data
def get_data():
    return load_data()

CLUSTER_NAMES = {
    0: "Festival Splurger",
    1: "Urban Fusion Millennial",
    2: "Daily Cotton Homemaker",
    3: "Premium Occasion Shopper",
    4: "Budget-Conscious Student",
    5: "The Gifter",
}

CLUSTER_OFFERS = {
    "Festival Splurger":        ("Festival_Sale 28% off salwar suits (Oct–Nov)", "WhatsApp broadcast + local retail"),
    "Urban Fusion Millennial":  ("Buy2Get1 Indo-western + Palazzo bundle", "Instagram Reels + D2C website"),
    "Daily Cotton Homemaker":   ("Free shipping on cotton kurtis above ₹799", "WhatsApp + word of mouth"),
    "Premium Occasion Shopper": ("Personalised member offer + early access", "Email + Brand website"),
    "Budget-Conscious Student": ("First order ₹150 off + COD available", "Instagram + social commerce"),
    "The Gifter":               ("Gift bundle: kurti + bedding set + gift wrap", "WhatsApp + Online marketplace"),
}

CLUSTER_COLORS = ["#1D9E75","#378ADD","#BA7517","#534AB7","#E24B4A","#888780"]

def get_cluster_features():
    return ["fashion_identity_enc","price_sensitivity_enc","brand_openness_enc",
            "online_purchase_confidence_enc","sustainability_consciousness_enc",
            "purchase_frequency_enc"] + BINARY_COLS

@st.cache_data
def run_clustering(n_clusters=6):
    df = get_data()
    df_enc = encode_features(df)
    feat_cols = get_cluster_features()
    feat_cols = [c for c in feat_cols if c in df_enc.columns]
    X = df_enc[feat_cols].fillna(0).values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc)
    sil = silhouette_score(X_sc, labels)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    return df, labels, sil, X_pca, feat_cols

def render():
    st.title("👥 Customer Clustering")
    st.caption("Who are Zaria's distinct customer personas? K-Means segmentation.")

    # Elbow + Silhouette
    st.markdown("<div class='section-hdr'>Optimal Number of Clusters</div>", unsafe_allow_html=True)

    with st.spinner("Computing elbow curve…"):
        df_raw = get_data()
        df_enc = encode_features(df_raw)
        feat_cols = get_cluster_features()
        feat_cols = [c for c in feat_cols if c in df_enc.columns]
        X = df_enc[feat_cols].fillna(0).values
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        inertias, sil_scores, k_range = [], [], range(2, 10)
        for k in k_range:
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl_tmp = km_tmp.fit_predict(X_sc)
            inertias.append(km_tmp.inertia_)
            sil_scores.append(silhouette_score(X_sc, lbl_tmp))

    c1,c2 = st.columns(2)
    with c1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertias,
                                       mode="lines+markers",
                                       line=dict(color="#1D9E75", width=2),
                                       marker=dict(size=8, color="#1D9E75"),
                                       name="Inertia"))
        fig_elbow.update_layout(title="Elbow Curve", xaxis_title="Number of Clusters (k)",
                                yaxis_title="Inertia", height=300,
                                margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig_elbow, use_container_width=True)

    with c2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(x=list(k_range), y=sil_scores,
                                     mode="lines+markers",
                                     line=dict(color="#378ADD", width=2),
                                     marker=dict(size=8, color="#378ADD"),
                                     name="Silhouette"))
        fig_sil.update_layout(title="Silhouette Score", xaxis_title="Number of Clusters (k)",
                              yaxis_title="Silhouette Score", height=300,
                              margin=dict(t=40,b=10,l=10,r=10))
        fig_sil.add_vline(x=6, line_dash="dash", line_color="red",
                          annotation_text="Optimal k=6")
        st.plotly_chart(fig_sil, use_container_width=True)

    n_clust = st.slider("Select number of clusters", 2, 9, 6)

    with st.spinner("Running K-Means…"):
        df, labels, sil, X_pca, feat_cols = run_clustering(n_clust)

    df = df.copy()
    df["cluster_id"] = labels
    df["cluster_name"] = df["cluster_id"].map(
        {i: CLUSTER_NAMES.get(i, f"Cluster {i}") for i in range(n_clust)})

    st.success(f"Silhouette Score: **{sil:.3f}** (higher is better, max 1.0)")

    # PCA Scatter
    st.markdown("<div class='section-hdr'>Cluster Visualisation (PCA 2D)</div>", unsafe_allow_html=True)
    pca_df = pd.DataFrame({"PC1": X_pca[:,0], "PC2": X_pca[:,1],
                           "Cluster": df["cluster_name"],
                           "Brand Openness": df["brand_openness"],
                           "Fashion Identity": df["fashion_identity"]})
    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                         color_discrete_sequence=CLUSTER_COLORS[:n_clust],
                         hover_data=["Brand Openness","Fashion Identity"],
                         title="Customer Clusters in 2D (PCA)",
                         opacity=0.7)
    fig_pca.update_traces(marker=dict(size=5))
    fig_pca.update_layout(height=450, margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig_pca, use_container_width=True)

    # Cluster profiles
    st.markdown("<div class='section-hdr'>Cluster Profiles & Marketing Playbook</div>", unsafe_allow_html=True)

    profile_cols = ["price_sensitivity","brand_openness","online_purchase_confidence",
                    "fashion_identity","purchase_frequency","zaria_interest_label"]
    cluster_profile = df.groupby("cluster_name")[profile_cols].agg(lambda x: x.mode()[0]).reset_index()

    for i, row in cluster_profile.iterrows():
        cname = row["cluster_name"]
        cnt   = (df["cluster_name"]==cname).sum()
        offer, channel = CLUSTER_OFFERS.get(cname, ("Custom offer","Mixed"))
        col_l, col_r = st.columns([2,1])
        with col_l:
            with st.expander(f"{'🟢🔵🟡🟣🔴⚫'[i % 6]} **{cname}** — {cnt} customers ({cnt/len(df)*100:.0f}%)"):
                c_a, c_b, c_c = st.columns(3)
                c_a.metric("Price Sensitivity", row["price_sensitivity"].replace("_"," "))
                c_b.metric("Brand Openness", row["brand_openness"].replace("_"," "))
                c_c.metric("Purchase Frequency", row["purchase_frequency"].replace("_"," "))
                c_d, c_e, c_f = st.columns(3)
                c_d.metric("Fashion Identity", row["fashion_identity"].replace("_"," "))
                c_e.metric("Online Confidence", row["online_purchase_confidence"].replace("_"," "))
                c_f.metric("Interest Label", row["zaria_interest_label"])
                st.markdown(f"**Recommended Offer:** {offer}")
                st.markdown(f"**Best Channel:** {channel}")

    # Cluster size bar
    st.markdown("<div class='section-hdr'>Cluster Size Distribution</div>", unsafe_allow_html=True)
    sz_df = df["cluster_name"].value_counts().reset_index()
    sz_df.columns = ["Cluster","Count"]
    fig_sz = px.bar(sz_df, x="Cluster", y="Count",
                    color="Cluster", color_discrete_sequence=CLUSTER_COLORS[:n_clust],
                    text="Count", title="Number of Customers per Cluster")
    fig_sz.update_traces(textposition="outside")
    fig_sz.update_layout(height=350, showlegend=False,
                         margin=dict(t=50,b=10,l=10,r=10), xaxis_title="")
    st.plotly_chart(fig_sz, use_container_width=True)

    # Discount strategy table
    st.markdown("<div class='section-hdr'>Discount Strategy per Cluster</div>", unsafe_allow_html=True)
    disc_tbl = df.groupby("cluster_name")["discount_preference"].agg(
        lambda x: x.value_counts().idxmax()).reset_index()
    disc_tbl.columns = ["Cluster","Top Discount Preference"]
    for _, r in disc_tbl.iterrows():
        offer, ch = CLUSTER_OFFERS.get(r["Cluster"],("—","—"))
        st.markdown(
            f"**{r['Cluster']}** → Preferred discount: `{r['Top Discount Preference']}` | "
            f"Zaria offer: _{offer}_ | Channel: _{ch}_")

    # Save cluster labels to session state for use in predictor
    st.session_state["cluster_labels"] = df["cluster_name"].values
    st.session_state["cluster_model_k"] = n_clust
