import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import plotly.express as px
except ImportError:
    import os
    os.system(f"{sys.executable} -m pip install plotly")
    import plotly.express as px
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Wine Classification & PCA Analytics",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Dashboard Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #722f37 0%, #4a1f24 100%);
        color: white;
        padding: 24px;
        border-radius: 15px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 8px 15px rgba(114, 47, 55, 0.2);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading & Model Training
# ---------------------------------------------------------
@st.cache_data
def load_data_and_train():
    # Load dataset
    df = pd.read_csv('Wine.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1]

    # Scaling
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    # PCA (2 Components for 2D visualization)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Classifier
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_pca, y)

    return df, feature_names, sc, pca, classifier, X_pca, y

try:
    df, feature_names, sc, pca, classifier, X_pca, y = load_data_and_train()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading 'Wine.csv': {e}")
    data_loaded = False

if data_loaded:
    # ---------------------------------------------------------
    # Header Section
    # ---------------------------------------------------------
    st.title("🍷 Wine Analytics & PCA Classification")
    st.write("An interactive web application for Principal Component Analysis (PCA) and Customer Segment prediction.")
    st.markdown("---")

    # ---------------------------------------------------------
    # Sidebar - Inputs
    # ---------------------------------------------------------
    st.sidebar.header("⚙️ Input Features")
    st.sidebar.write("Adjust feature values to predict the customer segment:")

    input_values = []
    for col in feature_names:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        
        val = st.sidebar.slider(
            label=f"{col}",
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            key=col
        )
        input_values.append(val)

    # ---------------------------------------------------------
    # Main Content Area
    # ---------------------------------------------------------
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("🎯 Prediction Result")
        
        # Preprocess single sample
        sample = np.array([input_values])
        sample_scaled = sc.transform(sample)
        sample_pca = pca.transform(sample_scaled)
        pred_class = classifier.predict(sample_pca)[0]
        pred_proba = classifier.predict_proba(sample_pca)[0]

        # Display Prediction Result Card
        st.markdown(f"""
            <div class="prediction-box">
                <h3 style="color: #ffffff; margin:0;">Predicted Segment: Customer Segment {pred_class}</h3>
                <p style="margin-top:10px; font-size:1.1em;">Confidence: {max(pred_proba)*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("##### 📊 Transformed Values (Principal Components):")
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="PC1 Value", value=f"{sample_pca[0][0]:.3f}")
        with c2:
            st.metric(label="PC2 Value", value=f"{sample_pca[0][1]:.3f}")

        st.markdown("##### 📈 Explained Variance Ratio:")
        exp_var = pca.explained_variance_ratio_
        st.info(f"PC1 accounts for **{exp_var[0]*100:.2f}%** of total variance, and PC2 accounts for **{exp_var[1]*100:.2f}%**.")

    with col_right:
        st.subheader("📉 PCA Interactive Scatter Plot")
        
        # Prepare DataFrame for Plotly
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['Segment'] = y.astype(str)

        # Plotly Interactive Chart
        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Segment',
            title="Wine Clusters & Current Sample in PCA Space",
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.75,
            height=450
        )

        # Add User Sample Highlight
        fig.add_scatter(
            x=[sample_pca[0][0]],
            y=[sample_pca[0][1]],
            mode='markers',
            marker=dict(size=16, color='black', symbol='star', line=dict(width=2, color='yellow')),
            name='Current Sample'
        )

        fig.update_layout(
            template="plotly_white",
            legend_title_text="Segment",
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # Raw Data Section
    # ---------------------------------------------------------
    with st.expander("📁 View Dataset (First 10 Rows)"):
        st.dataframe(df.head(10), use_container_width=True)





