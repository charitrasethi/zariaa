import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_data, encode_features, ORDINAL_MAPS

@st.cache_data
def get_data():
    return load_data()

def render():
    st.title("🔍 Diagnostic Analysis")
    st.caption("Why is it happening? Correlations, cross-tabs, and causal patterns.")

    df = get_data()
    df_enc = encode_features(df)

    # ── Correlation Heatmap ──────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Psychographic Correlation Heatmap</div>", unsafe_allow_html=True)

    corr_cols = [c + "_enc" for c in ORDINAL_MAPS if c + "_enc" in df_enc.columns]
    corr_labels = [c.replace("_enc","").replace("_"," ").title() for c in corr_cols]

    corr_matrix = df_enc[corr_cols].corr().round(2)
    corr_matrix.index = corr_labels
    corr_matrix.columns = corr_labels

    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size":9})
    ax.set_title("Correlation Matrix — Ordinal Features", fontsize=13, pad=10)
    plt.tight_layout()
    st.pyplot(fig_corr, use_container_width=True)
    plt.close()

    st.markdown("<div class='insight-box'>📌 Strong correlations between income, price sensitivity, and brand openness — these three together are the strongest predictors of customer value.</div>", unsafe_allow_html=True)

    # ── Cross-tab Explorer ───────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Cross-Tab Explorer</div>", unsafe_allow_html=True)
    cat_cols = ["region","city_tier","occupation","fashion_identity","price_sensitivity",
                "brand_openness","online_purchase_confidence","discovery_channel",
                "conversion_trigger","discount_preference","fabric_preference",
                "color_preference","purchase_frequency","zaria_interest_label"]

    c1,c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("Row variable", cat_cols, index=0)
    with c2:
        y_col = st.selectbox("Column variable", cat_cols, index=13)

    if x_col != y_col:
        cross = pd.crosstab(df[x_col], df[y_col], normalize="index").round(3) * 100
        fig_ct = px.imshow(cross, text_auto=".0f", aspect="auto",
                           color_continuous_scale="Greens",
                           title=f"{x_col.replace('_',' ').title()} × {y_col.replace('_',' ').title()} (row %)",
                           labels=dict(color="%"))
        fig_ct.update_layout(height=400, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_ct, use_container_width=True)
    else:
        st.warning("Please select two different variables.")

    # ── Brand Openness → Interest Funnel ────────────────────────────────────
    st.markdown("<div class='section-hdr'>Brand Openness → Zaria Interest Funnel</div>", unsafe_allow_html=True)

    bo_order = ["Very_Loyal_One_Brand","Loyal_But_Open","Multi_Brand_Shopper","No_Brand_Preference"]
    bo_int = df[df["zaria_interest_label"]=="Interested"]["brand_openness"].value_counts()
    bo_total = df["brand_openness"].value_counts()
    bo_df = pd.DataFrame({
        "Segment": bo_order,
        "Total": [bo_total.get(b,0) for b in bo_order],
        "Interested": [bo_int.get(b,0) for b in bo_order],
    })
    bo_df["Interest_Rate_%"] = (bo_df["Interested"] / bo_df["Total"] * 100).round(1)

    c3,c4 = st.columns(2)
    with c3:
        fig_bo = px.bar(bo_df, x="Segment", y="Interest_Rate_%",
                        color="Interest_Rate_%",
                        color_continuous_scale=["#FCEBEB","#1D9E75"],
                        text="Interest_Rate_%",
                        title="% Interested by Brand Openness Segment")
        fig_bo.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bo.update_layout(height=350, coloraxis_showscale=False,
                             margin=dict(t=50,b=10,l=10,r=10), xaxis_title="", xaxis_tickangle=-10)
        st.plotly_chart(fig_bo, use_container_width=True)

    with c4:
        fi_int = df.groupby("fashion_identity")["zaria_interest_label"].apply(
            lambda x: (x=="Interested").sum() / len(x) * 100).round(1).reset_index()
        fi_int.columns = ["Fashion Identity","Interest Rate %"]
        fi_int = fi_int.sort_values("Interest Rate %", ascending=True)
        fig_fi = px.bar(fi_int, x="Interest Rate %", y="Fashion Identity", orientation="h",
                        color="Interest Rate %", color_continuous_scale=["#FAEEDA","#854F0B"],
                        text="Interest Rate %",
                        title="Interest Rate by Fashion Identity")
        fig_fi.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_fi.update_layout(height=350, coloraxis_showscale=False,
                             margin=dict(t=50,b=10,l=10,r=10), yaxis_title="")
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Income vs Spend ──────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Income vs Estimated Annual Spend</div>", unsafe_allow_html=True)
    df_spend = df.dropna(subset=["monthly_income_band"])
    inc_order = ["Below_20K","20K_40K","40K_70K","70K_120K","Above_120K"]
    df_spend = df_spend[df_spend["monthly_income_band"].isin(inc_order)].copy()
    df_spend["monthly_income_band"] = pd.Categorical(df_spend["monthly_income_band"],
                                                      categories=inc_order, ordered=True)
    fig_sp = px.box(df_spend, x="monthly_income_band", y="estimated_annual_spend",
                    color="zaria_interest_label",
                    color_discrete_map={"Interested":"#1D9E75","Neutral":"#FFC107","Not_Interested":"#E24B4A"},
                    title="Annual Spend Distribution by Income Band & Interest Level",
                    labels={"monthly_income_band":"Income Band","estimated_annual_spend":"Est. Annual Spend (₹)"})
    fig_sp.update_layout(height=400, margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig_sp, use_container_width=True)

    # ── Price Sensitivity × Purchase Frequency ───────────────────────────────
    st.markdown("<div class='section-hdr'>Price Sensitivity × Purchase Frequency</div>", unsafe_allow_html=True)
    ps_pf = pd.crosstab(df["price_sensitivity"], df["purchase_frequency"], normalize="index").round(3)*100
    fig_pspf = px.imshow(ps_pf, text_auto=".0f", aspect="auto",
                         color_continuous_scale="Blues",
                         title="Purchase Frequency (%) by Price Sensitivity — row normalised",
                         labels=dict(color="%"))
    fig_pspf.update_layout(height=350, margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig_pspf, use_container_width=True)

    # ── Conversion Trigger Effectiveness ────────────────────────────────────
    st.markdown("<div class='section-hdr'>Conversion Trigger Effectiveness</div>", unsafe_allow_html=True)
    ct_eff = df.groupby("conversion_trigger")["zaria_interest_label"].apply(
        lambda x: (x=="Interested").sum()/len(x)*100).round(1).sort_values(ascending=False).reset_index()
    ct_eff.columns = ["Trigger","Interest_Rate_%"]
    fig_ct2 = px.bar(ct_eff, x="Interest_Rate_%", y="Trigger", orientation="h",
                     color="Interest_Rate_%", color_continuous_scale=["#E6F1FB","#185FA5"],
                     text="Interest_Rate_%",
                     title="% Interested by Conversion Trigger")
    fig_ct2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_ct2.update_layout(height=380, coloraxis_showscale=False,
                          margin=dict(t=50,b=10,l=10,r=10), yaxis_title="")
    st.plotly_chart(fig_ct2, use_container_width=True)
    st.markdown("<div class='insight-box'>💡 Use this chart to decide which conversion message to show each customer segment. High-interest triggers = first-message content for paid campaigns.</div>", unsafe_allow_html=True)
