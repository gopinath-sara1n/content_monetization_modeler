import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Ad Revenue Predictor",
    page_icon="📺",
    layout="wide"
)

# ---------------- LOAD ---------------- #
model = pickle.load(open(r"C:\Users\gopin\OneDrive\Documents\Data_Science\Streamlit Training files\Content Monetization Modeler\cmm\Scripts\model.pkl", "rb"))
columns = pickle.load(open(r"C:\Users\gopin\OneDrive\Documents\Data_Science\Streamlit Training files\Content Monetization Modeler\cmm\Scripts\columns.pkl", "rb"))

# ---------------- TITLE ---------------- #
st.title("📺 YouTube Ad Revenue Predictor")
st.markdown("Predict YouTube ad revenue based on engagement & watch behavior")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("📊 Input Parameters")

views = st.sidebar.number_input("Views", value=10000)
likes = st.sidebar.number_input("Likes", value=1000)
comments = st.sidebar.number_input("Comments", value=200)
watch_time = st.sidebar.number_input("Watch Time (minutes)", value=30000)
video_length = st.sidebar.number_input("Video Length (minutes)", value=15)
subscribers = st.sidebar.number_input("Subscribers", value=500000)

# ---------------- FEATURE ENGINEERING ---------------- #
retention_rate = watch_time / (views * video_length)
retention_rate = min(retention_rate, 2)

engagement_rate = (likes + comments) / views
log_subscribers = np.log1p(subscribers)

# ---------------- BUILD INPUT ---------------- #
input_data = {
    'views': views,
    'likes': likes,
    'comments': comments,
    'watch_time_minutes': watch_time,
    'video_length_minutes': video_length,
    'retention_rate': retention_rate,
    'log_subscribers': log_subscribers,
    'engagement_rate': engagement_rate
}

input_df = pd.DataFrame([input_data])
input_df = input_df[columns]

# ---------------- LAYOUT ---------------- #
col1, col2 = st.columns([2, 1])

# ---------------- MAIN PANEL ---------------- #
with col1:
    st.subheader("📈 Prediction")

    if st.button("🚀 Predict Revenue"):
        prediction = model.predict(input_df)[0]

        st.markdown(
            f"""
            <div style="
                background-color:#1f2937;
                padding:20px;
                border-radius:10px;
                text-align:center;
                color:white;
            ">
                <h2>💰 Predicted Revenue</h2>
                <h1 style="color:#22c55e;">${prediction:.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- SIDE INFO ---------------- #
with col2:
    st.subheader("📊 Insights")

    st.metric("Engagement Rate", f"{engagement_rate:.2f}")
    st.metric("Retention Rate", f"{retention_rate:.2f}")

    st.info("💡 Watch time is the most important factor driving revenue.")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")