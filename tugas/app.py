from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # agar matplotlib bisa jalan di server
import matplotlib.pyplot as plt
import io
import base64

# ---------------------------
# Load model & scaler
# ---------------------------
model = joblib.load("model_knn_regression.pkl")
scaler = joblib.load("scaler1.pkl")

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
# Inisialisasi Flask
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Route utama
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    pred = ""
    keterangan = ""
    plot_url = ""
    
    # Default input dari dataset
    t3_default = float(df['NO2'].values[-3]) if df is not None else 0.0
    t2_default = float(df['NO2'].values[-2]) if df is not None else 0.0
    t1_default = float(df['NO2'].values[-1]) if df is not None else 0.0

    if request.method == "POST":
        t3 = float(request.form["t3"])
        t2 = float(request.form["t2"])
        t1 = float(request.form["t1"])
        data_input = np.array([t3, t2, t1]).reshape(1, -1)
        data_scaled = scaler.transform(data_input)
        pred_value = model.predict(data_scaled)[0]
        pred = f"{pred_value:.8f}"
        keterangan, color = interpret_no2(pred_value)

        # ---------------------------
        # Buat plot baru setiap POST
        # ---------------------------
        fig, ax = plt.subplots(figsize=(8,4))
        if df is not None:
            ax.plot(df.index, df['NO2'], label='NO₂ harian', marker='o')
            # Highlight 3 hari terakhir
            ax.plot(df.index[-3:], df['NO2'].values[-3:], color='red', marker='o', linestyle='', label='3 hari terakhir')
            # Titik prediksi dengan warna sesuai keterangan
            ax.scatter(df.index[-1]+1, pred_value, color=color, s=100, label='Prediksi')
        ax.set_xlabel("Hari")
        ax.set_ylabel("NO₂ (ppm)")
        ax.set_title("Tren NO₂ Harian")
        ax.legend()

        # Convert plot ke base64 agar bisa ditampilkan di HTML
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close(fig)

    return render_template("index.html",
                           t3=t3_default,
                           t2=t2_default,
                           t1=t1_default,
                           pred=pred,
                           keterangan=keterangan,
                           plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
