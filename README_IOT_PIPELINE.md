# Instrucciones mínimas para iot_pipeline

Archivos incluidos:

- `start_iot_pipeline.bat` -- Script Windows para activar el entorno `tf-env` y ejecutar `iot_pipeline.py` en modo seguro (dry-run) por defecto.
- `tests/test_iot_pipeline.py` -- Tests unitarios rápidos que cubren `build_features` y `train_autoencoder`.

Cómo ejecutar los tests (PowerShell / Anaconda Prompt):

```powershell
# Activar el entorno conda (ajusta la ruta si tu conda está en otro sitio)
conda activate tf-env
python -m pip install -r requirements.txt  # opcional si aún no instalaste deps
python -m unittest discover -v
```

Cómo usar el .bat:

1. Edita `start_iot_pipeline.bat` y ajusta `CONDA_ROOT` si tu Anaconda está en otra carpeta.
2. Doble clic o ejecutar desde PowerShell para arrancar en modo dry-run.

Para ejecutar el listener MQTT real, elimina los flags `--dry-run --no-mqtt` en el .bat y asegura que InfluxDB y MQTT estén accesibles.

Crear tarea programada (Task Scheduler):

- Crear una tarea que ejecute `start_iot_pipeline.bat` con la cuenta deseada (por ejemplo, cuenta de servicio). Ajusta triggers y condiciones según necesites.
