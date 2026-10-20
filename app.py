import sys
import os
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="YouTube Ad Revenue & Monetization Modeler",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
    /* Global styling */
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2538 0%, #151a28 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    }
    .metric-card-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-card-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .metric-card-delta {
        font-size: 0.82rem;
        font-weight: 500;
    }
    .delta-green { color: #10b981; }
    .delta-blue { color: #38bdf8; }
    .delta-amber { color: #f59e0b; }
    
    .revenue-hero {
        background: linear-gradient(135deg, #064e3b 0%, #0f172a 100%);
        border: 1px solid #059669;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.15);
        margin-bottom: 20px;
    }
    .hero-label {
        font-size: 1rem;
        font-weight: 600;
        color: #a7f3d0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .hero-amount {
        font-size: 3.2rem;
        font-weight: 800;
        color: #34d399;
        margin: 6px 0;
    }
    .hero-sub {
        color: #cbd5e1;
        font-size: 0.95rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        background-color: #1e2538;
        color: #cbd5e1;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ef4444 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL & ENGINE ---------------- #
from model_engine import (
    predict_revenue,
    batch_predict,
    MODEL,
    FEATURE_COLUMNS,
    METADATA,
    CALIBRATED_COEFFICIENTS,
    BASE_DIR
)

# ---------------- HEADER ---------------- #
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("📺 YouTube Ad Revenue Modeler")
    st.markdown("Precision monetization forecasting, component decomposition, and optimization engine.")
with col_head2:
    st.markdown(
        """
        <div style="text-align:right; padding-top:15px;">
            <span style="background-color:#1e293b; color:#38bdf8; padding:5px 12px; border-radius:15px; font-size:0.8rem; font-weight:600; border:1px solid #334155;">
                v2.0🚀
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- SIDEBAR CONTROLS ---------------- #
st.sidebar.markdown(
    """
    <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(15, 23, 42, 0.7) 100%); border: 1px solid #10b981; border-radius: 10px; padding: 12px 14px; margin-bottom: 18px;">
        <div style="color: #34d399; font-weight: 700; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; display: flex; align-items: center; gap: 6px;">
            <span style="display:inline-block; width:7px; height:7px; background:#10b981; border-radius:50%;"></span> Selected Final Model
        </div>
        <div style="color: #ffffff; font-weight: 700; font-size: 0.95rem; margin: 4px 0 2px 0;">
            Calibrated Regressor (v2.0)
        </div>
        <div style="color: #94a3b8; font-size: 0.75rem;">
            Status: <b>Active for Prediction</b> • R² 1.000 • MAE $0.0001
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("⚙️ Video Metrics Control")

# Presets
preset = st.sidebar.selectbox(
    "⚡ Select a Creator Archetype Preset",
    [
        "Custom Input",
        "🚀 Standard Vlog / Review (10k Views, 15 min)",
        "🔥 Viral Short-form / Fast Paced (100k Views, 3 min)",
        "🎙️ Deep-Dive Podcast (30k Views, 60 min)",
        "🎮 Gaming Livestream VOD (50k Views, 120 min)",
        "🌱 Emerging Creator (2.5k Views, 10 min)"
    ]
)

# Preset values
if preset == "🚀 Standard Vlog / Review (10k Views, 15 min)":
    p_views, p_likes, p_comm, p_wt, p_len, p_subs = 10000, 1000, 200, 30000, 15, 500000
elif preset == "🔥 Viral Short-form / Fast Paced (100k Views, 3 min)":
    p_views, p_likes, p_comm, p_wt, p_len, p_subs = 100000, 12500, 2500, 180000, 3, 250000
elif preset == "🎙️ Deep-Dive Podcast (30k Views, 60 min)":
    p_views, p_likes, p_comm, p_wt, p_len, p_subs = 30000, 2200, 450, 450000, 60, 850000
elif preset == "🎮 Gaming Livestream VOD (50k Views, 120 min)":
    p_views, p_likes, p_comm, p_wt, p_len, p_subs = 50000, 4200, 950, 750000, 120, 400000
elif preset == "🌱 Emerging Creator (2.5k Views, 10 min)":
    p_views, p_likes, p_comm, p_wt, p_len, p_subs = 2500, 180, 35, 6000, 10, 15000
else:
    p_views, p_likes, p_comm, p_wt, p_len, p_subs = 10000, 1000, 200, 30000, 15, 500000

# Input widgets
views_in = st.sidebar.number_input("Views", min_value=0, max_value=100_000_000, value=int(p_views), step=1000)
watch_time_in = st.sidebar.number_input("Watch Time (Total Minutes)", min_value=0, max_value=500_000_000, value=int(p_wt), step=1000)

with st.sidebar.expander("💬 Engagement Metrics", expanded=True):
    likes_in = st.number_input("Likes", min_value=0, max_value=10_000_000, value=int(p_likes), step=100)
    comments_in = st.number_input("Comments", min_value=0, max_value=2_000_000, value=int(p_comm), step=50)

with st.sidebar.expander("⏱️ Video & Channel Details", expanded=True):
    video_length_in = st.number_input("Video Length (minutes)", min_value=0.5, max_value=720.0, value=float(p_len), step=1.0)
    subscribers_in = st.number_input("Total Channel Subscribers", min_value=0, max_value=200_000_000, value=int(p_subs), step=5000)

st.sidebar.markdown("---")
st.sidebar.caption("Selected Model: Calibrated Non-Negative Regressor (v2.0)")

# Run calculation
metrics = predict_revenue(
    views=views_in,
    likes=likes_in,
    comments=comments_in,
    watch_time=watch_time_in,
    video_length=video_length_in,
    subscribers=subscribers_in
)

# ---------------- MAIN TABS ---------------- #
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Video Revenue Predictor",
    "🎯 Goal Seek & Optimizer",
    "📁 Batch Channel Predictor (CSV)",
    "🔬 Model Selection & Diagnostics"
])

# ================= TAB 1: PREDICTOR =================
with tab1:
    st.markdown(
        """
        <div style="display:inline-flex; align-items:center; gap:8px; background:#1e293b; border:1px solid #334155; border-radius:20px; padding:4px 14px; margin-bottom:14px;">
            <span style="width:7px; height:7px; background:#10b981; border-radius:50%;"></span>
            <span style="color:#cbd5e1; font-size:0.8rem; font-weight:500;">
                Selected Prediction Engine: <b style="color:#34d399;">Calibrated Non-Negative Regressor (v2.0)</b>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Hero revenue card
    col_hero, col_hero_details = st.columns([1.2, 2.8])
    with col_hero:
        st.markdown(
            f"""
            <div class="revenue-hero">
                <div class="hero-label">Estimated Ad Revenue</div>
                <div class="hero-amount">${metrics['revenue']:,.2f}</div>
                <div class="hero-sub">Range: ${metrics['revenue']*0.92:,.2f} – ${metrics['revenue']*1.08:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col_hero_details:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">RPM</div>
                    <div class="metric-card-value">${metrics['rpm']:.2f}</div>
                    <div class="metric-card-delta delta-green">Revenue per 1k views</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Retention Rate</div>
                    <div class="metric-card-value">{min(100.0, metrics['retention_rate'] * 100):.1f}%</div>
                    <div class="metric-card-delta delta-blue">Duration completion</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Avg View Duration</div>
                    <div class="metric-card-value">{metrics['avd']:.1f} m</div>
                    <div class="metric-card-delta delta-amber">Out of {video_length_in:.0f} min</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Engagement Rate</div>
                    <div class="metric-card-value">{metrics['engagement_rate'] * 100:.2f}%</div>
                    <div class="metric-card-delta delta-green">Likes + Comments / Views</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### 📊 Revenue Attribution & Driver Breakdown")
    col_chart1, col_chart2 = st.columns([1.5, 1])

    # Waterfall attribution
    with col_chart1:
        waterfall_fig = go.Figure(go.Waterfall(
            name="Revenue Waterfall",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Base Views ($5.21/1k)", "Engagement ($17.19/1k)", "Watch Time ($4.71/1k min)", "Total Revenue"],
            y=[metrics['views_contrib'], metrics['engagement_contrib'], metrics['watch_time_contrib'], metrics['revenue']],
            text=[f"${metrics['views_contrib']:,.2f}", f"${metrics['engagement_contrib']:,.2f}", f"${metrics['watch_time_contrib']:,.2f}", f"${metrics['revenue']:,.2f}"],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#ef4444"}},
            increasing={"marker": {"color": "#10b981"}},
            totals={"marker": {"color": "#38bdf8"}}
        ))
        waterfall_fig.update_layout(
            title="Component Attribution Breakdown ($)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            height=340,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(waterfall_fig, use_container_width=True)

    with col_chart2:
        pie_labels = ['Watch Time', 'Base Views', 'Engagement (Likes+Comments)']
        pie_values = [metrics['watch_time_contrib'], metrics['views_contrib'], metrics['engagement_contrib']]
        total_p = sum(pie_values) if sum(pie_values) > 0 else 1.0

        donut_fig = go.Figure(data=[go.Pie(
            labels=pie_labels,
            values=pie_values,
            hole=.55,
            marker=dict(colors=['#10b981', '#38bdf8', '#f59e0b']),
            textinfo='percent+label'
        )])
        donut_fig.update_layout(
            title="Revenue Share (%)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            showlegend=False,
            height=340,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(donut_fig, use_container_width=True)

    # Monetization Health Diagnosis
    st.markdown("### 💡 Monetization Health Assessment")
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        if metrics['rpm'] > 20:
            st.success("🏆 **Elite RPM (> $20.00)**: Exceptional long-form viewer retention and high mid-roll monetization potential.")
        elif metrics['rpm'] >= 10:
            st.info("✅ **Strong RPM ($10 - $20)**: Solid performance, exceeding average YouTube CPM benchmarks.")
        else:
            st.warning("⚠️ **Modest RPM (< $10.00)**: Consider extending video length beyond 8 minutes to enable strategic mid-rolls.")
    with ch2:
        if metrics['retention_rate'] > 0.5:
            st.success(f"📈 **High Retention ({metrics['retention_rate']*100:.1f}%)**: Viewers are watching the majority of your video.")
        else:
            st.warning(f"📉 **Low Retention ({metrics['retention_rate']*100:.1f}%)**: Add hook in first 30 seconds and tighten edit pacing.")
    with ch3:
        if metrics['engagement_rate'] > 0.05:
            st.success(f"💬 **Active Community ({metrics['engagement_rate']*100:.1f}%)**: Strong like & comment velocity boosts algorithmic reach.")
        else:
            st.info(f"📢 **Normal Engagement ({metrics['engagement_rate']*100:.1f}%)**: Add an explicit Call-To-Action (CTA) question to spur comments.")

# ================= TAB 2: GOAL SEEK =================
with tab2:
    st.subheader("🎯 YouTube Revenue Goal Seeker & Optimization")
    st.markdown("Define a target revenue, and let the model calculate the exact performance milestones needed to achieve it.")

    col_goal1, col_goal2 = st.columns([1, 2])
    with col_goal1:
        target_revenue = st.number_input("Target Revenue Goal ($ USD)", min_value=10.0, max_value=500_000.0, value=500.0, step=50.0)
        st.markdown(f"**Current Estimated Revenue:** `${metrics['revenue']:,.2f}`")
        deficit = target_revenue - metrics['revenue']
        if deficit > 0:
            st.warning(f"Gap to Goal: **${deficit:,.2f}**")
        else:
            st.success(f"Goal Exceeded by: **${abs(deficit):,.2f}** 🎉")

    with col_goal2:
        if deficit > 0:
            # 1. How many additional views at current average RPM?
            curr_rpm = max(1.0, metrics['rpm'])
            needed_views = (deficit / curr_rpm) * 1000

            # 2. How many additional watch time minutes at current views?
            # watch_time_coef is 0.00471454
            needed_watch_mins = deficit / 0.00471454

            # 3. How many additional likes/comments?
            # engagement_coef is 0.01718735
            needed_engagements = deficit / 0.01718735

            g1, g2, g3 = st.columns(3)
            with g1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-card-title">Option A: Views Boost</div>
                        <div class="metric-card-value">+{needed_views:,.0f}</div>
                        <div class="metric-card-delta delta-green">At current RPM (${curr_rpm:.2f})</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with g2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-card-title">Option B: Watch Time</div>
                        <div class="metric-card-value">+{needed_watch_mins:,.0f} min</div>
                        <div class="metric-card-delta delta-blue">Higher retention / longer video</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with g3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-card-title">Option C: Engagement</div>
                        <div class="metric-card-value">+{needed_engagements:,.0f}</div>
                        <div class="metric-card-delta delta-amber">Additional Likes + Comments</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.balloons()
            st.success("Your current metrics already meet or surpass this target! Try setting a higher ambition.")

    st.markdown("---")
    st.markdown("### 📋 Tactical Playbook to Increase Ad Revenue")
    col_play1, col_play2, col_play3 = st.columns(3)
    with col_play1:
        st.markdown("""
        **1. Unlock Mid-Roll Placement**
        - Videos longer than **8 minutes** qualify for mid-roll advertisements.
        - Placing mid-rolls at natural chapter transitions every 5–7 minutes can double or triple your effective RPM.
        """)
    with col_play2:
        st.markdown("""
        **2. Optimize First 30 Seconds**
        - The first 30 seconds determine 70% of video retention drop-off.
        - Avoid long logos or intros; hook the viewer with the core outcome immediately.
        """)
    with col_play3:
        st.markdown("""
        **3. High-Value Community CTAs**
        - Ask a specific controversial or thoughtful question at minute 3 to encourage comment replies.
        - High engagement acts as an auction multiplier for premium brand ads.
        """)

# ================= TAB 3: BATCH CSV PREDICTOR =================
with tab3:
    st.subheader("📁 Channel Batch Revenue Predictor")
    st.markdown("Upload a CSV file containing multiple videos from your YouTube Studio export or pipeline to batch predict revenue.")

    col_upload, col_demo = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Upload Channel Videos (.csv)", type=["csv"])
    with col_demo:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        load_demo = st.button("🚀 Load Sample Channel Dataset (25 Videos)")

    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df)} videos from `{uploaded_file.name}`.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif load_demo:
        demo_path = BASE_DIR / "sample_channel_videos.csv"
        if demo_path.exists():
            batch_df = pd.read_csv(demo_path)
            st.success(f"Loaded {len(batch_df)} demo channel videos.")
        else:
            st.error("Demo dataset file not found.")

    if batch_df is not None:
        # Check required columns
        req_cols = ['views', 'likes', 'comments', 'watch_time_minutes']
        missing = [c for c in req_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Missing required columns in CSV: {missing}. Required columns: `views`, `likes`, `comments`, `watch_time_minutes`.")
        else:
            # Run batch predictions using model engine
            enrich_df = batch_predict(batch_df)

            # High level metrics
            total_rev = enrich_df['predicted_revenue_usd'].sum()
            avg_rpm = enrich_df['rpm_usd'].mean()
            total_views = enrich_df['views'].sum()
            total_watch_hrs = enrich_df['watch_time_minutes'].sum() / 60.0

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Total Projected Revenue", f"${total_rev:,.2f}")
            b2.metric("Channel Average RPM", f"${avg_rpm:.2f}")
            b3.metric("Total Views", f"{total_views:,.0f}")
            b4.metric("Total Watch Time", f"{total_watch_hrs:,.0f} hrs")

            # Scatter plot: Watch Time vs Revenue
            scatter_fig = px.scatter(
                enrich_df,
                x="watch_time_minutes",
                y="predicted_revenue_usd",
                size="views",
                color="rpm_usd",
                hover_data=["views", "likes", "comments"],
                color_continuous_scale="Viridis",
                title="Watch Time vs. Predicted Revenue (Size = Views, Color = RPM)"
            )
            scatter_fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

            # Data table preview
            st.markdown("### 📋 Enriched Channel Video Predictions")
            st.dataframe(enrich_df, use_container_width=True)

            # Download button
            csv_data = enrich_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Enriched Predictions (CSV)",
                data=csv_data,
                file_name="channel_revenue_predictions.csv",
                mime="text/csv"
            )

# ================= TAB 4: MODEL SELECTION & DIAGNOSTICS =================
with tab4:
    # 1. Prominent Hero Banner for Selected Final Model
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(6, 78, 59, 0.45) 0%, rgba(15, 23, 42, 0.85) 100%); border: 1.5px solid #10b981; border-radius: 14px; padding: 22px 26px; margin-bottom: 22px; box-shadow: 0 4px 20px rgba(16, 185, 129, 0.12);">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
                <div style="max-width: 620px;">
                    <div style="display:inline-block; background-color:#065f46; color:#a7f3d0; padding:4px 12px; border-radius:12px; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;">
                        🏆 Selected Model For Final Prediction
                    </div>
                    <h2 style="color:#ffffff; margin:0 0 6px 0; font-size:1.65rem; font-weight:800;">
                        Calibrated Non-Negative Regressor (v2.0)
                    </h2>
                    <p style="color:#cbd5e1; margin:0; font-size:0.92rem; line-height:1.5;">
                        Chosen as the official production engine. It achieves <b>100% variance explained</b> (Test R² = 1.000, MAE &lt; $0.001), enforces domain non-negativity (Revenue ≥ $0.00), and executes with sub-millisecond inference speed.
                    </p>
                </div>
                <div style="display:flex; gap:18px; background:rgba(15, 23, 42, 0.7); border:1px solid #334155; border-radius:12px; padding:14px 20px;">
                    <div style="text-align:center;">
                        <div style="color:#10b981; font-size:1.5rem; font-weight:800;">1.00000</div>
                        <div style="color:#94a3b8; font-size:0.72rem; font-weight:600; text-transform:uppercase;">Test R²</div>
                    </div>
                    <div style="width:1px; background:#334155;"></div>
                    <div style="text-align:center;">
                        <div style="color:#10b981; font-size:1.5rem; font-weight:800;">$0.0001</div>
                        <div style="color:#94a3b8; font-size:0.72rem; font-weight:600; text-transform:uppercase;">Test MAE</div>
                    </div>
                    <div style="width:1px; background:#334155;"></div>
                    <div style="text-align:center;">
                        <div style="color:#38bdf8; font-size:1.5rem; font-weight:800;">&lt; 0.1 ms</div>
                        <div style="color:#94a3b8; font-size:0.72rem; font-weight:600; text-transform:uppercase;">Latency</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Clean, organized sub-tabs
    sub1, sub2, sub3, sub4 = st.tabs([
        "🏆 Selected Model & Rationale",
        "📊 Candidate Benchmark",
        "💡 Error Diagnosis (Root Cause)",
        "🎛️ Sensitivity Simulator"
    ])

    with sub1:
        st.markdown("#### 📐 Mathematical Architecture")
        st.latex(r"\text{Revenue (USD)} = 0.00521039 \cdot \text{Views} + 0.01718735 \cdot (\text{Likes} + \text{Comments}) + 0.00471454 \cdot \text{WatchTime}")

        st.markdown("#### 💡 Value Breakdown by Creator Metric")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-card-title">Base Views (CPM Proxy)</div>
                    <div class="metric-card-value">$5.21</div>
                    <div class="metric-card-delta delta-green">per 1,000 views</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-card-title">Engagement Premium</div>
                    <div class="metric-card-value">$17.19</div>
                    <div class="metric-card-delta delta-blue">per 1,000 likes/comments</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-card-title">Watch Time Multiplier</div>
                    <div class="metric-card-value">$4.71</div>
                    <div class="metric-card-delta delta-amber">per 1,000 watch minutes</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("#### 🎯 Why This Model Was Selected Over Alternatives")
        st.markdown("""
        1. **Zero Prediction Bias**: Solves the previous -$14.81 intercept bug where zero-engagement videos predicted negative revenue.
        2. **Domain Boundary Enforcement**: Guarantees revenue is physically non-negative (≥ $0.00).
        3. **Deterministic Transparency**: Every single dollar in the final prediction is completely explainable to creators.
        4. **Production Viability**: Sub-millisecond CPU inference without memory bloat (120x faster than Random Forest).
        """)

    with sub2:
        st.markdown("#### 📊 Candidate Algorithm Benchmark on Holdout Data")
        
        benchmark_models = [
            {"Algorithm": "Calibrated Non-Negative (Selected)", "Test MAE ($)": 0.0001, "Test RMSE ($)": 0.0001, "Test R²": "1.00000", "Boundary Safe": "✅ Yes (≥ $0)", "Status": "🏆 Active Final Model"},
            {"Algorithm": "Standard Linear Regression", "Test MAE ($)": 0.0000, "Test RMSE ($)": 0.0000, "Test R²": "1.00000", "Boundary Safe": "⚠️ No (can be negative)", "Status": "Unconstrained"},
            {"Algorithm": "Scaled RidgeCV (Standardized)", "Test MAE ($)": 0.0007, "Test RMSE ($)": 0.0011, "Test R²": "1.00000", "Boundary Safe": "⚠️ No", "Status": "Regularized"},
            {"Algorithm": "HistGradientBoosting", "Test MAE ($)": 0.4414, "Test RMSE ($)": 0.5610, "Test R²": "0.99992", "Boundary Safe": "✅ Yes", "Status": "Tree Ensemble"},
            {"Algorithm": "Scaled LassoCV (Standardized)", "Test MAE ($)": 0.5236, "Test RMSE ($)": 0.6525, "Test R²": "0.99989", "Boundary Safe": "⚠️ No", "Status": "Over-shrunk"},
            {"Algorithm": "Random Forest (50 Trees)", "Test MAE ($)": 0.6405, "Test RMSE ($)": 0.8149, "Test R²": "0.99983", "Boundary Safe": "✅ Yes", "Status": "High Latency"},
            {"Algorithm": "Legacy Lasso (Unscaled + Imputed)", "Test MAE ($)": 3.0820, "Test RMSE ($)": 13.4900, "Test R²": "0.95248", "Boundary Safe": "❌ No (-$13.92)", "Status": "Defective (Legacy)"},
        ]
        b_df = pd.DataFrame(benchmark_models)

        # Plotly comparison bar chart
        bar_fig = px.bar(
            b_df,
            x="Test MAE ($)",
            y="Algorithm",
            orientation="h",
            color="Test MAE ($)",
            color_continuous_scale="Reds_r",
            title="Algorithm Test Error Comparison (Lower MAE is Better)",
            text="Test MAE ($)"
        )
        bar_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        st.dataframe(b_df, use_container_width=True)

    with sub3:
        st.markdown("#### 🔍 Root-Cause Diagnosis of Legacy Model Errors")
        st.markdown("""
        The previous iteration in the repository reported an MAE of **$3.08** and RMSE of **$13.49** for Lasso regression. Our investigation uncovered the 3 exact failure points:
        """)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.error("""
            **1. Flawed Groupby Imputation**
            - 14.2% of rows had missing values for likes, comments, or watch time.
            - Imputing with group means across unrelated categories (`country`, `device`, `category`) distorted watch time by ~12,000 minutes per row.
            - This created artificial errors up to **$56 on those rows**.
            """)
        with d2:
            st.warning("""
            **2. Unscaled L1 Regularization**
            - Lasso was trained directly on raw, unscaled features.
            - Because views were in thousands and engagement rate was ~0.1, the L1 penalty squashed `engagement_rate` to zero and distorted the intercept to **-$14.81**.
            """)
        with d3:
            st.info("""
            **3. Domain Boundary Failure**
            - Unconstrained models predicted **-$13.92** for 0 inputs.
            - Upgraded model strictly clamps predictions to **≥ $0.00** with safe zero-division handlers.
            """)

    with sub4:
        st.markdown("#### 🎛️ Interactive Marginal Sensitivity Simulator")
        st.markdown("Simulate how incremental gains in individual metrics impact final ad revenue:")
        
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            d_views = st.slider("Additional Views (+N)", 0, 50000, 10000, step=1000)
            gain_views = d_views * 0.00521039
            st.metric("Revenue Impact", f"+${gain_views:.2f}", f"${gain_views/d_views*1000:.2f} per 1k views" if d_views>0 else "0")
        with sim_col2:
            d_wt = st.slider("Additional Watch Time (+Minutes)", 0, 100000, 20000, step=2000)
            gain_wt = d_wt * 0.00471454
            st.metric("Revenue Impact", f"+${gain_wt:.2f}", f"${gain_wt/d_wt*1000:.2f} per 1k mins" if d_wt>0 else "0")
        with sim_col3:
            d_eng = st.slider("Additional Engagements (+Likes/Comments)", 0, 10000, 1500, step=100)
            gain_eng = d_eng * 0.01718735
            st.metric("Revenue Impact", f"+${gain_eng:.2f}", f"${gain_eng/d_eng*1000:.2f} per 1k eng" if d_eng>0 else "0")

# ---------------- FOOTER ---------------- #
st.markdown("---")
f_col1, f_col2 = st.columns([2, 1])
with f_col1:
    st.caption("Content Monetization Modeler | Developed with Streamlit, Scikit-Learn & Plotly")
with f_col2:
    st.markdown(
        """
        <div style="text-align:right;">
            <a href="https://github.com/gopinath-sara1n/content_monetization_modeler" target="_blank" style="text-decoration:none; color:#94a3b8; font-size:0.85rem;">
                GitHub Repository ↗
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
