import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import iot_pipeline

# Crear datos sintéticos de ejemplo (3 horas, 1min frecuencia)
now = datetime.utcnow()
times = pd.date_range(now - pd.Timedelta(hours=3), periods=180, freq='1min')
values = np.sin(np.linspace(0, 10, len(times))) * 100 + 200

df = pd.DataFrame({'time': times, 'value': values, 'device': ['plug01'] * len(times)})

# Sustituir fetch_influx_data para devolver los datos sintéticos
def fake_fetch(hours=24, device=None):
    return df

iot_pipeline.fetch_influx_data = fake_fetch

# Reducir entrenamiento del autoencoder para prueba rápida
orig_train_ae = iot_pipeline.train_autoencoder

def fast_train_autoencoder(X_train, X_test, epochs=5, batch_size=32, model_path=None):
    # llamar al original pero con menos epochs y batch size
    return orig_train_ae(X_train, X_test, epochs=5, batch_size=32, model_path=model_path or 'outputs/ae_test.h5')

iot_pipeline.train_autoencoder = fast_train_autoencoder

# Ejecutar pipeline completo (usará los datos sintéticos y el entrenamiento rápido)
iot_pipeline.run_full_pipeline(device_id='plug01')

print('run_full_pipeline() de prueba finalizado.')
