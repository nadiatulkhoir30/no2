import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------
# Load model & scaler
# ---------------------------
@st.cache_resource
def load_model_scaler():
    model = joblib.load("model_knn_regression.pkl")
    scaler = joblib.load("scaler1.pkl")
    return model, scaler

model, scaler = load_model_scaler()

# ---------------------------
# Fungsi interpretasi
# ---------------------------
def interpret_no2(value):
    if value < 2e-5:
        return "Baik ✅", "green"
    elif value < 3e-5:
        return "Sedang ⚠️", "orange"
    else:
        return "Buruk ❌", "red"

# ---------------------------
# Load data CSV
# ---------------------------
try:
    df = pd.read_csv("timeseries.csv")
    df['NO2'] = df['NO2'].astype(float)
except:
    df = None

# ---------------------------
# Header aplikasi
# ---------------------------
st.set_page_config(page_title="Prediksi NO₂", layout="centered")
st.markdown(
    """
    <div style="text-align:center">
        <h1>🌬️ Prediksi NO₂ Menggunakan KNN Regression</h1>
        <p>Masukkan tiga nilai NO₂ terakhir untuk memprediksi NO₂ hari ini.</p>
    </div>
    """, unsafe_allow_html=True
)

# ---------------------------
# Input nilai NO2
# ---------------------------
col1, col2, col3 = st.columns(3)
t3_default = float(df['NO2'].values[-3]) if df is not None else 0.0
t2_default = float(df['NO2'].values[-2]) if df is not None else 0.0
t1_default = float(df['NO2'].values[-1]) if df is not None else 0.0

with col1:
    t3 = st.number_input("NO₂(t-3)", value=t3_default, format="%.8f")
with col2:
    t2 = st.number_input("NO₂(t-2)", value=t2_default, format="%.8f")
with col3:
    t1 = st.number_input("NO₂(t-1)", value=t1_default, format="%.8f")

# ---------------------------
# Tombol prediksi
# ---------------------------
if st.button("Prediksi NO₂"):
    data_input = np.array([t3, t2, t1]).reshape(1, -1)
    data_scaled = scaler.transform(data_input)
    pred_value = model.predict(data_scaled)[0]
    keterangan, color = interpret_no2(pred_value)

    st.markdown(f"### Hasil Prediksi: **{pred_value:.8f}**")
    st.markdown(f"<h4 style='color:{color}'>Keterangan kualitas udara: {keterangan}</h4>", unsafe_allow_html=True)

    # ---------------------------
    # Plot tren NO2
    # ---------------------------
    fig, ax = plt.subplots(figsize=(8,4))
    if df is not None:
        ax.plot(df.index, df['NO2'], label='NO₂ harian', marker='o')
        ax.plot(df.index[-3:], df['NO2'].values[-3:], color='red', marker='o', linestyle='', label='3 hari terakhir')
        ax.scatter(df.index[-1]+1, pred_value, color=color, s=100, label='Prediksi')
    ax.set_xlabel("Hari")
    ax.set_ylabel("NO₂ (ppm)")
    ax.set_title("Tren NO₂ Harian")
    ax.legend()
    st.pyplot(fig)
