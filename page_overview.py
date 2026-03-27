import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_loader import load_data

@st.cache_data
def get_data():
    return load_data()

def render():
    st.title("🏠 Zaria Fashion — Executive Overview")
    st.caption("Founder's morning dashboard · Pan-India customer survey · 1,200 respondents")

    df = get_data()

    interested = (df["zaria_interest_label"] == "Interested").sum()
    neutral    = (df["zaria_interest_label"] == "Neutral").sum()
    not_int    = (df["zaria_interest_label"] == "Not_Interested").sum()
    top_region = df["region"].value_counts().idxmax()
    top_persona_raw = df.groupby(["fashion_identity"])["zaria_interest_label"].apply(
        lambda x: (x=="Interested").sum()).idxmax()
    avg_spend  = df["estimated_annual_spend"].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-val'>{len(df):,}</div>
            <div class='metric-lbl'>Total Respondents</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-val' style='color:#1D9E75'>{interested:,}</div>
            <div class='metric-lbl'>Interested in Zaria</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-val' style='color:#FFC107'>{neutral:,}</div>
            <div class='metric-lbl'>Neutral</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-val' style='color:#E24B4A'>{not_int:,}</div>
            <div class='metric-lbl'>Not Interested</div></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-val'>₹{avg_spend/1000:.1f}K</div>
            <div class='metric-lbl'>Avg Annual Spend</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.markdown("<div class='section-hdr'>Zaria Interest Distribution</div>", unsafe_allow_html=True)
        fig = px.pie(
            names=["Interested","Neutral","Not Interested"],
            values=[interested, neutral, not_int],
            color_discrete_sequence=["#1D9E75","#FFC107","#E24B4A"],
            hole=0.45,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-hdr'>Interest by Region</div>", unsafe_allow_html=True)
        reg_int = df[df["zaria_interest_label"]=="Interested"]["region"].value_counts().reset_index()
        reg_int.columns = ["Region","Count"]
        fig2 = px.bar(reg_int, x="Count", y="Region", orientation="h",
                      color="Count", color_continuous_scale=["#9FE1CB","#1D9E75"],
                      text="Count")
        fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300,
                           coloraxis_showscale=False, yaxis_title="", xaxis_title="Interested Customers")
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.markdown("<div class='section-hdr'>City Tier Breakdown</div>", unsafe_allow_html=True)
        tier_df = df["city_tier"].value_counts().reset_index()
        tier_df.columns = ["Tier","Count"]
        fig3 = px.bar(tier_df, x="Tier", y="Count",
                      color="Tier", color_discrete_sequence=["#1D9E75","#378ADD","#BA7517","#888780"],
                      text="Count")
        fig3.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300,
                           showlegend=False, xaxis_title="", yaxis_title="Respondents")
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("<div class='section-hdr'>Fashion Identity Mix</div>", unsafe_allow_html=True)
        fi_df = df["fashion_identity"].value_counts().reset_index()
        fi_df.columns = ["Identity","Count"]
        colors = ["#1D9E75","#378ADD","#BA7517","#534AB7","#E24B4A","#888780"]
        fig4 = px.bar(fi_df, x="Identity", y="Count", color="Identity",
                      color_discrete_sequence=colors, text="Count")
        fig4.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300,
                           showlegend=False, xaxis_title="", yaxis_title="Count",
                           xaxis_tickangle=-20)
        fig4.update_traces(textposition="outside")
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        st.markdown("<div class='section-hdr'>Top Conversion Triggers</div>", unsafe_allow_html=True)
        ct_df = df["conversion_trigger"].value_counts().reset_index()
        ct_df.columns = ["Trigger","Count"]
        fig5 = px.bar(ct_df, x="Count", y="Trigger", orientation="h",
                      color="Count", color_continuous_scale=["#B5D4F4","#185FA5"],
                      text="Count")
        fig5.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300,
                           coloraxis_showscale=False, yaxis_title="", xaxis_title="Count")
        fig5.update_traces(textposition="outside")
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-hdr'>Founder's Key Insights</div>", unsafe_allow_html=True)
    pct_int = interested/len(df)*100
    top_trigger = df["conversion_trigger"].value_counts().idxmax()
    top_channel = df["preferred_shopping_channel"].value_counts().idxmax()
    top_inc = df["monthly_income_band"].value_counts().idxmax()

    i1,i2,i3,i4 = st.columns(4)
    with i1:
        st.markdown(f"<div class='insight-box'>🎯 <b>{pct_int:.0f}%</b> of respondents show interest in Zaria — strong product-market fit signal for a new brand.</div>", unsafe_allow_html=True)
    with i2:
        st.markdown(f"<div class='insight-box'>🔑 Top conversion trigger is <b>{top_trigger.replace('_',' ')}</b> — prioritise this in launch messaging.</div>", unsafe_allow_html=True)
    with i3:
        st.markdown(f"<div class='insight-box'>🛒 Most customers prefer <b>{top_channel.replace('_',' ')}</b> — invest here first for distribution.</div>", unsafe_allow_html=True)
    with i4:
        st.markdown(f"<div class='insight-box'>💰 Dominant income band is <b>{top_inc.replace('_',' ')}</b> — price products between ₹800–₹2,500 for widest reach.</div>", unsafe_allow_html=True)
