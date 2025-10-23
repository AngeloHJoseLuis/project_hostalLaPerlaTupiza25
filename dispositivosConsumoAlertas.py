# Este script crea una plantilla de base de datos para un hostal
# con hojas para dispositivos, consumo y alertas.
# Asegúrate de tener instalada la librería pandas
# Puedes instalarla con: pip install pandas openpyxl

import pandas as pd

# Crear datos 
data = {
    'ID Dispositivo': [1, 2, 3],
    'Nombre': ['Aire Acondicionado', 'Televisor', 'Hervidor'],
    'Ubicación': ['Hab. 1', 'Hab. 2', 'Cocina'],
    'Tipo': ['Climatización', 'Entretenimiento', 'Electrodoméstico']
}

consumo = {
    'ID Consumo': [1, 2, 3],
    'ID Dispositivo': [1, 2, 3],
    'Fecha': ['2025-08-17 14:00:00', '2025-08-17 15:00:00', '2025-08-17 16:00:00'],
    'Consumo (W)': [480.0, 190.0, 350.0]
}

alertas = {
    'ID Alerta': [1, 2],
    'ID Consumo': [1, 3],
    'Exceso (%)': [32.0, 28.0],
    'Mensaje': ['Exceso detectado en aire acondicionado', 'Alerta por consumo anormal en hervidor'],
    'Acción sugerida': ['Verificar uso', 'Desconectar si no está en uso'],
    'Fecha': ['2025-08-17 14:05:00', '2025-08-17 16:10:00']
}

# Crear DataFrames
df_dispositivos = pd.DataFrame(data)
df_consumo = pd.DataFrame(consumo)
df_alertas = pd.DataFrame(alertas)

# Guardar en Excel (especificar engine openpyxl)
with pd.ExcelWriter("plantilla_base_datos_hostal.xlsx", engine="openpyxl") as writer:
    df_dispositivos.to_excel(writer, sheet_name="Dispositivos", index=False)
    df_consumo.to_excel(writer, sheet_name="Consumo", index=False)
    df_alertas.to_excel(writer, sheet_name="Alertas", index=False)
    
# Mensaje de éxito
print("Plantilla de base de datos creada exitosamente: plantilla_base_datos_hostal.xlsx")
