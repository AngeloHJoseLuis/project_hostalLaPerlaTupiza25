# Este script escribe un punto de datos en InfluxDB
# Asegúrate de que InfluxDB esté corriendo y que las credenciales sean correctas
# Requiere la librería influxdb-client
# Puedes instalarla con: pip install influxdb-client
# Importa las librerías necesarias
from influxdb_client import InfluxDBClient, Point, WritePrecision # pyright: ignore[reportMissingImports]
from influxdb_client.client.write_api import SYNCHRONOUS # pyright: ignore[reportMissingImports]

token = "tu_token"
org = "tu_organizacion"
bucket = "hostal_perla_tupiza"

client = InfluxDBClient(url="http://localhost:8086", token=token)
write_api = client.write_api(write_options=SYNCHRONOUS)

point = Point("lecturas_energia") \
    .tag("habitacion", "Hab. 1") \
    .tag("dispositivo", "TV") \
    .field("corriente", 1.2) \
    .field("voltaje", 220) \
    .field("potencia", 264) \
    .time("2025-08-17T14:32:00Z")

write_api.write(bucket=bucket, org=org, record=point)
client.close()
