import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from data_loader import load_data, BINARY_COLS

@st.cache_data
def get_data():
    return load_data()

def render():
    st.title("📊 Descriptive Analysis")
    st.caption("What is the current state of Zaria's potential market?")

    df = get_data()

    # Sidebar filters
    with st.sidebar:
        st.markdown("### Filters")
        regions = st.multiselect("Region", df["region"].unique().tolist(),
                                  default=df["region"].unique().tolist())
        tiers = st.multiselect("City Tier", df["city_tier"].unique().tolist(),
                                default=df["city_tier"].unique().tolist())
        ages = st.multiselect("Age Group", df["age_group"].unique().tolist(),
                               default=df["age_group"].unique().tolist())

    mask = (df["region"].isin(regions) &
            df["city_tier"].isin(tiers) &
            df["age_group"].isin(ages))
    df = df[mask].copy()
    st.caption(f"Showing **{len(df):,}** respondents after filters")

    # ── Demographics ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Demographics</div>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)

    with c1:
        age_df = df["age_group"].value_counts().reindex(
            ["Under_18","18-24","25-34","35-44","45-54","55+"], fill_value=0).reset_index()
        age_df.columns = ["Age","Count"]
        fig = px.bar(age_df, x="Age", y="Count", color="Count",
                     color_continuous_scale=["#9FE1CB","#1D9E75"], text="Count",
                     title="Age Distribution")
        fig.update_layout(height=300, coloraxis_showscale=False,
                          margin=dict(t=40,b=10,l=10,r=10), xaxis_title="")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        occ_df = df["occupation"].value_counts().reset_index()
        occ_df.columns = ["Occupation","Count"]
        fig2 = px.pie(occ_df, names="Occupation", values="Count",
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      title="Occupation Mix", hole=0.35)
        fig2.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        inc_order = ["Below_20K","20K_40K","40K_70K","70K_120K","Above_120K"]
        inc_df = df["monthly_income_band"].value_counts().reindex(inc_order, fill_value=0).reset_index()
        inc_df.columns = ["Income","Count"]
        fig3 = px.bar(inc_df, x="Income", y="Count", color="Count",
                      color_continuous_scale=["#B5D4F4","#185FA5"], text="Count",
                      title="Income Distribution")
        fig3.update_layout(height=300, coloraxis_showscale=False,
                           margin=dict(t=40,b=10,l=10,r=10), xaxis_title="", xaxis_tickangle=-15)
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Psychographics ───────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Psychographic Profiles</div>", unsafe_allow_html=True)
    c4,c5 = st.columns(2)

    with c4:
        fi_df = df["fashion_identity"].value_counts().reset_index()
        fi_df.columns = ["Identity","Count"]
        fig4 = px.bar(fi_df, x="Count", y="Identity", orientation="h",
                      color="Count", color_continuous_scale=["#F4C0D1","#993556"],
                      text="Count", title="Fashion Identity")
        fig4.update_layout(height=320, coloraxis_showscale=False,
                           margin=dict(t=40,b=10,l=10,r=10), yaxis_title="")
        fig4.update_traces(textposition="outside")
        st.plotly_chart(fig4, use_container_width=True)

    with c5:
        ps_order = ["Extremely_Price_Sensitive","Price_Conscious","Balanced","Quality_Over_Price","Price_Irrelevant"]
        ps_df = df["price_sensitivity"].value_counts().reindex(ps_order, fill_value=0).reset_index()
        ps_df.columns = ["Sensitivity","Count"]
        fig5 = px.bar(ps_df, x="Count", y="Sensitivity", orientation="h",
                      color="Count", color_continuous_scale=["#FAC775","#854F0B"],
                      text="Count", title="Price Sensitivity")
        fig5.update_layout(height=320, coloraxis_showscale=False,
                           margin=dict(t=40,b=10,l=10,r=10), yaxis_title="")
        fig5.update_traces(textposition="outside")
        st.plotly_chart(fig5, use_container_width=True)

    c6,c7 = st.columns(2)
    with c6:
        bo_order = ["Very_Loyal_One_Brand","Loyal_But_Open","Multi_Brand_Shopper","No_Brand_Preference"]
        bo_df = df["brand_openness"].value_counts().reindex(bo_order, fill_value=0).reset_index()
        bo_df.columns = ["Openness","Count"]
        fig6 = px.funnel(bo_df, x="Count", y="Openness",
                         title="Brand Openness Funnel",
                         color_discrete_sequence=["#1D9E75"])
        fig6.update_layout(height=320, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig6, use_container_width=True)

    with c7:
        sc_df = df["sustainability_consciousness"].value_counts().reset_index()
        sc_df.columns = ["Sustainability","Count"]
        fig7 = px.pie(sc_df, names="Sustainability", values="Count",
                      color_discrete_sequence=["#173404","#3B6D11","#97C459","#C0DD97"],
                      title="Sustainability Consciousness", hole=0.4)
        fig7.update_layout(height=320, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig7, use_container_width=True)

    # ── Product Ownership Heatmap ────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Product Ownership Heatmap</div>", unsafe_allow_html=True)
    prod_by_region = df.groupby("region")[BINARY_COLS].mean().round(3) * 100
    prod_by_region.columns = [c.replace("owns_","").replace("_"," ").title() for c in prod_by_region.columns]
    fig8 = px.imshow(prod_by_region,
                     color_continuous_scale="Greens",
                     text_auto=".0f",
                     aspect="auto",
                     title="% of customers owning each product — by Region",
                     labels=dict(color="Ownership %"))
    fig8.update_layout(height=350, margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig8, use_container_width=True)

    # ── Shopping Behaviour ───────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Shopping Behaviour</div>", unsafe_allow_html=True)
    c8,c9,c10 = st.columns(3)

    with c8:
        pf_order = ["Monthly_Plus","Every_2_3_Months","Every_6_Months","Festival_Only","Rarely"]
        pf_df = df["purchase_frequency"].value_counts().reindex(pf_order,fill_value=0).reset_index()
        pf_df.columns = ["Frequency","Count"]
        fig9 = px.bar(pf_df, x="Frequency", y="Count", color="Count",
                      color_continuous_scale=["#9FE1CB","#085041"],
                      text="Count", title="Purchase Frequency")
        fig9.update_layout(height=300, coloraxis_showscale=False,
                           margin=dict(t=40,b=10,l=10,r=10), xaxis_tickangle=-20, xaxis_title="")
        fig9.update_traces(textposition="outside")
        st.plotly_chart(fig9, use_container_width=True)

    with c9:
        ch_df = df["preferred_shopping_channel"].value_counts().reset_index()
        ch_df.columns = ["Channel","Count"]
        fig10 = px.pie(ch_df, names="Channel", values="Count",
                       color_discrete_sequence=px.colors.qualitative.Pastel,
                       title="Preferred Shopping Channel", hole=0.35)
        fig10.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig10, use_container_width=True)

    with c10:
        dc_df = df["discovery_channel"].value_counts().reset_index()
        dc_df.columns = ["Channel","Count"]
        fig11 = px.bar(dc_df, x="Count", y="Channel", orientation="h",
                       color="Count", color_continuous_scale=["#CECBF6","#3C3489"],
                       text="Count", title="Discovery Channel")
        fig11.update_layout(height=300, coloraxis_showscale=False,
                            margin=dict(t=40,b=10,l=10,r=10), yaxis_title="")
        fig11.update_traces(textposition="outside")
        st.plotly_chart(fig11, use_container_width=True)

    # ── Fabric & Colour ──────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Fabric & Colour Preferences</div>", unsafe_allow_html=True)
    c11,c12 = st.columns(2)
    with c11:
        fab_df = df["fabric_preference"].value_counts().reset_index()
        fab_df.columns = ["Fabric","Count"]
        fig12 = px.bar(fab_df, x="Fabric", y="Count", color="Fabric",
                       color_discrete_sequence=px.colors.qualitative.Safe,
                       text="Count", title="Fabric Preference")
        fig12.update_layout(height=300, showlegend=False,
                            margin=dict(t=40,b=10,l=10,r=10), xaxis_title="")
        fig12.update_traces(textposition="outside")
        st.plotly_chart(fig12, use_container_width=True)

    with c12:
        col_df = df["color_preference"].value_counts().reset_index()
        col_df.columns = ["Color","Count"]
        palette = {"Pastels":"#F4C0D1","Bright_Vibrant":"#E24B4A","Jewel_Tones":"#534AB7",
                   "Neutrals":"#B4B2A9","Dark_Tones":"#2C2C2A","Bold_Prints":"#BA7517"}
        fig13 = px.bar(col_df, x="Color", y="Count",
                       color="Color", color_discrete_map=palette,
                       text="Count", title="Colour Preference")
        fig13.update_layout(height=300, showlegend=False,
                            margin=dict(t=40,b=10,l=10,r=10), xaxis_title="")
        fig13.update_traces(textposition="outside")
        st.plotly_chart(fig13, use_container_width=True)

    # ── Raw data table ───────────────────────────────────────────────────────
    with st.expander("📋 View Raw Data"):
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button("Download filtered data", df.to_csv(index=False).encode(),
                           "zaria_filtered.csv", "text/csv")
