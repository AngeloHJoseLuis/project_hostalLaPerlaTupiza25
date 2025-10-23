"""
iot_pipeline.py
Script ejemplo para:
- Recibir datos de Smart Plugs (MQTT o HTTP)
- Guardar en InfluxDB (v2)
- Entrenar modelos ML: SVM y Autoencoder
- Evaluar y generar gráficos (consumo real vs predicho; tendencia diaria)

Requiere: paho-mqtt, influxdb-client, pandas, scikit-learn, tensorflow, matplotlib, joblib
"""

import os
import time
import json
import threading
import logging
import argparse
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# MQTT
import paho.mqtt.client as mqtt

# InfluxDB v2 client
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

# ML
from sklearn.svm import OneClassSVM, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Autoencoder (Keras)
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------
# CONFIGURACIÓN (editar)
# -----------------------
# MQTT broker (ej.: broker local o cloud)
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "smartplug/+/telemetry"   # ejemplo: smartplug/{id}/telemetry

# InfluxDB v2
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "TU_TOKEN_AQUI"
INFLUX_ORG = "TU_ORG"
INFLUX_BUCKET = "energy_bucket"

# Carpeta salida figuras y modelos
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Variables de muestreo y ventana
SAMPLE_INTERVAL = 5  # segundos (coincide con lo configurado en los Smart Plugs)
DATA_WINDOW_HOURS = 24 * 7  # cuánto historial traer para ML (ej. 7 días)

# -----------------------
# Conexión InfluxDB
# -----------------------
client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()

# -----------------------
# MQTT callbacks
# -----------------------
def on_connect(client_mqtt, userdata, flags, rc):
    print("MQTT conectado con rc:", rc)
    client_mqtt.subscribe(MQTT_TOPIC)
    print("Suscrito al tópico:", MQTT_TOPIC)

def on_message(client_mqtt, userdata, msg):
    try:
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        # Esperamos datos con al menos: {"id":"plug01","ts":"2025-08-17T14:20:00Z","current":0.27,"voltage":230.1}
        device_id = data.get("id") or data.get("device_id") or (msg.topic.split("/")[1] if "/" in msg.topic else "unknown")
        ts = data.get("ts") or data.get("timestamp") or datetime.utcnow().isoformat()
        current_a = float(data.get("current", 0.0))
        voltage = float(data.get("voltage", 0.0)) if data.get("voltage") else 230.0
        power_w = data.get("power") or current_a * voltage

        # Escribir en InfluxDB
        p = Point("energy_measurement") \
            .tag("device", device_id) \
            .field("current_a", float(current_a)) \
            .field("voltage", float(voltage)) \
            .field("power_w", float(power_w)) \
            .time(ts)
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)
        print(f"Guardado: {device_id} {ts} {power_w:.2f} W")

    except Exception as e:
        print("Error procesando mensaje MQTT:", e)

# -----------------------
# Iniciar MQTT listener en hilo
# -----------------------
def start_mqtt_listener():
    client_mqtt = mqtt.Client()
    client_mqtt.on_connect = on_connect
    client_mqtt.on_message = on_message
    client_mqtt.connect(MQTT_BROKER, MQTT_PORT, 60)
    client_mqtt.loop_forever()

# -----------------------
# Recuperar datos de InfluxDB como DataFrame
# -----------------------
def fetch_influx_data(hours=24, device=None):
    now = datetime.utcnow()
    start = (now - timedelta(hours=hours)).isoformat() + "Z"
    end = now.isoformat() + "Z"
    device_filter = f' and r["device"] == "{device}"' if device else ""
    query = f'''
    from(bucket:"{INFLUX_BUCKET}")
        |> range(start: {start}, stop: {end})
        |> filter(fn: (r) => r._measurement == "energy_measurement")
        |> filter(fn: (r) => r._field == "power_w")
    '''
    # Note: for device filtering, adapt for Flux syntax; simpler: read more and filter in pandas
    tables = query_api.query_data_frame(query)
    if tables.empty:
        return pd.DataFrame()
    # Flux return: multiple rows; try to transform
    df = tables
    # normalize column names
    if "_time" in df.columns:
        df = df.rename(columns={"_time":"time","_value":"value","device":"device"})
    # select useful cols
    cols = [c for c in df.columns if c in ("time","value","device")]
    if not cols:
        return pd.DataFrame()
    df = df[cols].dropna()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    return df

# -----------------------
# Preprocesado y creación de features
# -----------------------
def build_features(df, resample='5s'):
    # df: time, value, device
    if df.empty:
        return df
    df = df.set_index('time').resample(resample).mean().interpolate()
    df['power_w'] = df['value']
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    # rolling stats
    df['power_mean_1m'] = df['power_w'].rolling('1min').mean().bfill()
    df['power_std_5m'] = df['power_w'].rolling('5min').std().fillna(0)
    df = df.drop(columns=['value'], errors='ignore')
    return df

# -----------------------
# Entrenar y evaluar SVM (clasificación binaria simple: anómalo/no)
# Para demo usamos OneClassSVM o SVC si etiquetas disponibles
# -----------------------
def train_evaluate_svm(X_train, X_test, y_train=None, y_test=None, model_path=os.path.join(OUT_DIR,'svm_model.joblib')):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if y_train is None:
        # Unsupervised anomaly detection
        svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
        svm.fit(X_train_s)
        y_pred = svm.predict(X_test_s)
        # OneClassSVM returns 1 (inlier) or -1 (outlier)
        y_pred = np.where(y_pred==1, 0, 1)  # 1->inlier => label 0 normal ; -1 -> 1 anomalía
        report = None
        acc = None
    else:
        clf = SVC(kernel='rbf', C=1.0, probability=True)
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        report = classification_report(y_test, y_pred, output_dict=False)
        acc = accuracy_score(y_test, y_pred)
        joblib.dump((clf, scaler), model_path)
    return y_pred, report, acc

# -----------------------
# Autoencoder con Keras
# -----------------------
def build_autoencoder(input_dim, encoding_dim=8):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation="relu")(input_layer)
    encoded = layers.Dense(int(encoding_dim/2), activation="relu")(encoded)
    decoded = layers.Dense(encoding_dim, activation="relu")(encoded)
    decoded = layers.Dense(input_dim, activation="linear")(decoded)
    autoencoder = models.Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(X_train, X_test, epochs=30, batch_size=64, model_path=None):
    input_dim = X_train.shape[1]
    ae = build_autoencoder(input_dim, encoding_dim=max(4, input_dim//2))
    history = ae.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, X_test), verbose=1)
    # Guardar en formato nativo Keras (.keras) por defecto
    if model_path is None:
        model_path = os.path.join(OUT_DIR, 'ae_model.keras')
    else:
        # si se pasó .h5, sugerimos usar .keras pero respetamos la elección del usuario
        if model_path.endswith('.h5'):
            model_path = model_path.rsplit('.', 1)[0] + '.keras'
    ae.save(model_path)
    return ae, history

# -----------------------
# Funciones para graficar
# -----------------------
def plot_real_vs_pred(df_real, df_pred, out_file=os.path.join(OUT_DIR,'consumo_real_vs_predicho.png')):
    # df_real: DataFrame with time index and 'power_w'
    # df_pred: same index with 'predicted_power'
    plt.figure(figsize=(12,4))
    plt.plot(df_real.index, df_real['power_w'], label='Consumo real', linewidth=1.2)
    plt.plot(df_pred.index, df_pred['predicted_power'], label='Consumo predicho', linewidth=1.0, linestyle='--')
    plt.xlabel('Tiempo')
    plt.ylabel('Potencia (W)')
    plt.title('Comparación Consumo real vs Consumo predicho')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print("Guardado gráfico:", out_file)

def plot_daily_trend(df, out_file=os.path.join(OUT_DIR,'tendencia_diaria.png')):
    # df: time-indexed series
    df2 = df.copy()
    df2['hour'] = df2.index.hour
    trend = df2.groupby('hour')['power_w'].mean()
    plt.figure(figsize=(10,4))
    plt.plot(trend.index, trend.values, marker='o')
    plt.xlabel('Hora del día')
    plt.ylabel('Potencia promedio (W)')
    plt.title('Tendencia promedio de consumo diario')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print("Guardado gráfico:", out_file)

# -----------------------
# EJEMPLO de flujo completo (para correr manualmente)
# -----------------------
def run_full_pipeline(device_id=None):
    # 1) recuperar datos
    df_raw = fetch_influx_data(hours=24*3, device=device_id)  # 3 días ejemplo
    if df_raw.empty:
        print("No hay datos en Influx. Ejecuta el listener MQTT y genera datos.")
        return

    # si trae columna 'value' -> renombrar
    if 'value' in df_raw.columns and 'power_w' not in df_raw.columns:
        df_raw['power_w'] = df_raw['value']

    # reindex por time
    df = df_raw[['time','power_w']].rename(columns={'time':'time'}).set_index('time')
    df = df.resample('5s').mean().interpolate()  # estandarizar frecuencia

    # construir features simples
    df_feat = build_features(df.reset_index().rename(columns={'index':'time','power_w':'value'}).set_index('time').reset_index())
    df_feat = df_feat.dropna()

    # ML dataset (usar columnas numéricas)
    features = ['power_w','power_mean_1m','power_std_5m','hour']
    df_feat = df_feat.dropna()
    X = df_feat[features].values

    # split
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=42)

    # Entrenar autoencoder
    scaler_ae = StandardScaler()
    X_train_s = scaler_ae.fit_transform(X_train)
    X_test_s = scaler_ae.transform(X_test)
    # Si se quiere un modo rápido para pruebas, train_autoencoder puede recibir un model_path None
    ae, history = train_autoencoder(X_train_s, X_test_s, epochs=20, batch_size=128)

    # obtener reconstrucción y error
    X_test_rec = ae.predict(X_test_s)
    mse = np.mean(np.square(X_test_s - X_test_rec), axis=1)

    # definir umbral simple (ej. media + 3*std)
    thresh = mse.mean() + 3*mse.std()
    anomalies = mse > thresh

    print(f"Autoencoder umbral: {thresh:.6f}, anomalías detectadas: {anomalies.sum()} / {len(mse)}")

    # preparar series predicha (a modo de ejemplo: usar la reconstrucción del primer feature)
    # aquí generamos una serie de predicción "mock" para visualizar
    df_pred = df.copy()
    # simular predicción usando media móvil como aproximación
    df_pred['predicted_power'] = df_pred['power_w'].rolling('1min').mean().bfill()

    # graficar
    plot_real_vs_pred(df, df_pred)
    plot_daily_trend(df)

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='IoT pipeline runner')
    parser.add_argument('--no-mqtt', action='store_true', help='No iniciar listener MQTT')
    parser.add_argument('--dry-run', action='store_true', help='Ejecutar pipeline en modo prueba con datos sintéticos')
    parser.add_argument('--run-pipeline', action='store_true', help='Ejecutar run_full_pipeline() inmediatamente (útil con --dry-run)')
    args = parser.parse_args()

    if not args.no_mqtt and not args.dry_run:
        mqtt_thread = threading.Thread(target=start_mqtt_listener, daemon=True)
        mqtt_thread.start()
        logging.info('Listener MQTT iniciado. Espera a que lleguen datos o ejecuta run_full_pipeline() cuando haya registros.')
    else:
        logging.info('Modo sin MQTT activado.' if args.no_mqtt else 'Dry-run activado.')

    if args.dry_run and args.run_pipeline:
        # generar datos sintéticos y ejecutar run_full_pipeline
        logging.info('Generando datos sintéticos para dry-run y ejecutando run_full_pipeline()')
        # crear datos sintéticos simples
        now = datetime.utcnow()
        times = pd.date_range(now - pd.Timedelta(hours=3), periods=180, freq='1min')
        values = (np.sin(np.linspace(0, 10, len(times))) * 100) + 200
        df = pd.DataFrame({'time': times, 'value': values, 'device': ['plug01'] * len(times)})

        # sustituir temporalmente la función de fetch
        def _fake_fetch(hours=24, device=None):
            return df
        fetch_influx_data_backup = fetch_influx_data
        try:
            globals()['fetch_influx_data'] = _fake_fetch
            run_full_pipeline(device_id='plug01')
        finally:
            globals()['fetch_influx_data'] = fetch_influx_data_backup

    # Mantener la ejecución si se inició el listener
    if not args.dry_run and not args.no_mqtt:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info('Detenido por usuario.')
