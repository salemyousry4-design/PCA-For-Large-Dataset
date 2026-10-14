# 🍷 Wine Analytics & PCA Classification Dashboard

An advanced, end-to-end Machine Learning web application utilizing **Principal Component Analysis (PCA)** for dimensionality reduction and interactive customer segment/wine classification. Deployed live on Streamlit Cloud.

## 🌟 Live Demo
Explore the live interactive web application here: [Wine PCA Streamlit App](https://pca-for-large-dataset-git-6judfarlhqxcu8qc5kdhfo.streamlit.app)

---

## 🚀 Key Features
* **Interactive Control Panel (Sidebar):** Adjust chemical properties of wine (Alcohol, Malic Acid, Ash, Magnesium, etc.) via responsive sliders.
* **PCA Dimensionality Reduction:** Projects high-dimensional chemical features down to principal components (`PC1` & `PC2`).
* **Interactive Scatter Plot:** Real-time visual mapping of wine clusters and the user's custom input sample.
* **Instant Prediction:** Predicts the target wine segment/class with confidence estimation.

---

## 🛠️ Tech Stack & Libraries
* **Python** (Programming Language)
* **Scikit-Learn** (PCA, StandardScaler, Machine Learning Model)
* **Streamlit** (Interactive Dashboard UI & Cloud Deployment)
* **Plotly / Matplotlib** (Data Visualization & Scatter Plots)
* **Joblib & NumPy** (Data processing and serialization)

---

## 📂 Project Structure
```text
PCA-For-Large-Dataset/
│
├── app.py                      # Main Streamlit web application script
├── principal_component_analysis.py # PCA & Model training script
├── Wine.csv                    # Wine chemical dataset
├── requirements.txt            # Project dependencies
├── runtime.txt                 # Python runtime configuration
└── principal_component_analysis.ipynb # Jupyter notebook for analysis
