# Este script inserta un registro de consumo de energía en una base de datos MySQL
# Asegúrate de que MySQL esté corriendo y que las credenciales sean correctas
# Requiere la librería mysql-connector-python
# Puedes instalarla con: pip install mysql-connector-python
# Importa las librerías necesarias
import mysql.connector # pyright: ignore[reportMissingImports]
from datetime import datetime

conn = mysql.connector.connect(
    host="localhost",
    user="tu_usuario",
    password="tu_clave",
    database="hostal_energia"
)

cursor = conn.cursor()

# Insertar nuevo registro
sql = "INSERT INTO Consumo (id_dispositivo, fecha, consumo_watts) VALUES (%s, %s, %s)"
data = (1, datetime.now(), 132.5)

cursor.execute(sql, data)
conn.commit()

print("Registro insertado")
cursor.close()
conn.close()
