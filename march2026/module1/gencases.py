import pandas as pd
import numpy as np
import random
import os
import argparse

# Fijamos la semilla para que los resultados sean reproducibles
np.random.seed(42)
random.seed(42)

NUM_ROWS = 1000

# ==========================================
# 1. FUNCIÓN PARA INTRODUCIR RUIDO Y OUTLIERS
# ==========================================
def introducir_ruido(df, perc_nulos, perc_outliers, columnas_numericas):
    """
    Introduce valores nulos (NaN) y outliers multiplicando por factores extremos.
    """
    df_ruido = df.copy()
    
    # Introducir Nulos
    for col in df_ruido.columns:
        mask_nulos = np.random.rand(len(df_ruido)) < perc_nulos
        df_ruido.loc[mask_nulos, col] = np.nan
        
    # Introducir Outliers (solo en numéricas)
    for col in columnas_numericas:
        mask_outliers = np.random.rand(len(df_ruido)) < perc_outliers
        valid_mask = mask_outliers & df_ruido[col].notna()
        factores_validos = np.random.choice([10, 50, -10, -50], size=valid_mask.sum())
        df_ruido.loc[valid_mask, col] = df_ruido.loc[valid_mask, col] * factores_validos
            
    return df_ruido

# ==========================================
# 2. FUNCIONES GENERADORAS DE DATASETS (CASOS)
# ==========================================

def generar_caso1_carreteras():
    data = {
        'id_tramo': [f"TR-{i}" for i in range(NUM_ROWS)],
        'trafico_diario': np.random.randint(500, 20000, NUM_ROWS),
        'indice_desgaste_firme': np.random.uniform(1.0, 10.0, NUM_ROWS).round(2),
        'inversion_mantenimiento_eur': np.random.uniform(5000, 100000, NUM_ROWS).round(2),
        'necesita_reparacion': np.random.choice(['Sí', 'No'], NUM_ROWS, p=[0.3, 0.7])
    }
    return pd.DataFrame(data), ['trafico_diario', 'indice_desgaste_firme', 'inversion_mantenimiento_eur']

def generar_caso2_rrss():
    data = {
        'id_post': [f"POST-{i}" for i in range(NUM_ROWS)],
        'red_social': np.random.choice(['Instagram', 'TikTok', 'YouTube'], NUM_ROWS),
        'visualizaciones': np.random.randint(1000, 500000, NUM_ROWS),
        'likes': np.random.randint(50, 50000, NUM_ROWS),
        'comentarios': np.random.randint(0, 1000, NUM_ROWS),
        'tiempo_retencion_seg': np.random.uniform(10.0, 180.0, NUM_ROWS).round(1)
    }
    df = pd.DataFrame(data)
    df['likes'] = (df['visualizaciones'] * np.random.uniform(0.05, 0.15, NUM_ROWS)).astype(int)
    return df, ['visualizaciones', 'likes', 'comentarios', 'tiempo_retencion_seg']

def generar_caso3_estructuras():
    data = {
        'id_pieza': [f"PZ-{i}" for i in range(NUM_ROWS)],
        'temp_fundicion_C': np.random.normal(1500, 50, NUM_ROWS).round(1),
        'presion_molde_bar': np.random.normal(300, 20, NUM_ROWS).round(1),
        'num_porosidades_detectadas': np.random.randint(0, 15, NUM_ROWS),
        'estado_calidad': np.random.choice(['Aprobado', 'Rechazado'], NUM_ROWS, p=[0.85, 0.15])
    }
    return pd.DataFrame(data), ['temp_fundicion_C', 'presion_molde_bar', 'num_porosidades_detectadas']

def generar_caso4_web():
    data = {
        'id_sesion': [f"SESS-{i}" for i in range(NUM_ROWS)],
        'origen_trafico': np.random.choice(['Organico', 'Directo', 'Referido', 'Social'], NUM_ROWS),
        'paginas_vistas': np.random.randint(1, 15, NUM_ROWS),
        'tiempo_en_sitio_seg': np.random.uniform(5.0, 600.0, NUM_ROWS).round(1),
        'convirtio_compra': np.random.choice([1, 0], NUM_ROWS, p=[0.08, 0.92])
    }
    return pd.DataFrame(data), ['paginas_vistas', 'tiempo_en_sitio_seg', 'convirtio_compra']

def generar_caso5_recreativas():
    data = {
        'id_maquina': [f"MAC-{i}" for i in range(NUM_ROWS)],
        'tipo_local': np.random.choice(['Bar', 'Salon_Juego', 'Casino', 'Centro_Comercial'], NUM_ROWS),
        'tipo_maquina': np.random.choice(['Tragaperras', 'Arcade', 'Pinball', 'Apuestas'], NUM_ROWS),
        'recaudacion_mensual_eur': np.random.normal(2500, 800, NUM_ROWS).round(2),
        'incidencias_tecnicas': np.random.randint(0, 5, NUM_ROWS)
    }
    df = pd.DataFrame(data)
    df['recaudacion_mensual_eur'] = df['recaudacion_mensual_eur'].apply(lambda x: max(x, 0))
    return df, ['recaudacion_mensual_eur', 'incidencias_tecnicas']

def generar_caso6_baterias():
    data = {
        'id_bateria': [f"BAT-{i}" for i in range(NUM_ROWS)],
        'capacidad_nominal_ah': np.random.choice([100, 200, 500, 1000], NUM_ROWS),
        'ciclos_carga_actuales': np.random.randint(10, 2000, NUM_ROWS),
        'temperatura_media_operacion_C': np.random.normal(25, 10, NUM_ROWS).round(1),
        'salud_bateria_SOH_perc': np.random.uniform(60.0, 100.0, NUM_ROWS).round(1)
    }
    df = pd.DataFrame(data)
    df['salud_bateria_SOH_perc'] = 100 - (df['ciclos_carga_actuales'] / 2000) * 30 + np.random.normal(0, 2, NUM_ROWS)
    df['salud_bateria_SOH_perc'] = df['salud_bateria_SOH_perc'].clip(0, 100).round(2)
    return df, ['capacidad_nominal_ah', 'ciclos_carga_actuales', 'temperatura_media_operacion_C', 'salud_bateria_SOH_perc']

def generar_caso7_aseguradoras():
    data = {
        'id_siniestro': [f"SIN-{i}" for i in range(NUM_ROWS)],
        'tipo_seguro': np.random.choice(['Hogar', 'Pyme', 'Comunidad'], NUM_ROWS),
        'gremio_necesario': np.random.choice(['Fontaneria', 'Electricidad', 'Pintura', 'Albañileria'], NUM_ROWS),
        'coste_estimado_eur': np.random.uniform(100, 3000, NUM_ROWS).round(2),
        'dias_hasta_resolucion': np.random.randint(1, 45, NUM_ROWS)
    }
    df = pd.DataFrame(data)
    df['coste_real_eur'] = df['coste_estimado_eur'] * np.random.uniform(0.9, 1.3, NUM_ROWS)
    df['coste_real_eur'] = df['coste_real_eur'].round(2)
    return df, ['coste_estimado_eur', 'dias_hasta_resolucion', 'coste_real_eur']

# ==========================================
# 3. BLOQUE PRINCIPAL DE EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    # Configurar los argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Generador de Datasets Sintéticos con ruido.")
    parser.add_argument("--savedir", type=str, default="CSVs", 
                        help="Directorio donde se guardarán los archivos CSV (por defecto: 'CSVs')")
    
    # Parsear los argumentos. 
    # Usamos parse_known_args() en lugar de parse_args() para que no falle si se ejecuta 
    # directamente en una celda de Jupyter/Colab que a veces inyecta argumentos ocultos.
    args, unknown = parser.parse_known_args()
    
    directorio_salida = args.savedir
    
    # Crear el directorio si no existe
    os.makedirs(directorio_salida, exist_ok=True)
    print(f"Carpeta de destino configurada: ./{directorio_salida}/")

    casos = {
        "Andres_Carreteras": generar_caso1_carreteras,
        "Javier_RRSS": generar_caso2_rrss,
        "Carlos_EstructurasMet": generar_caso3_estructuras,
        "Amparo_AnaliticaWeb": generar_caso4_web,
        "Josep_Recreativas": generar_caso5_recreativas,
        "Fernando_Baterias": generar_caso6_baterias,
        "Juanjo_Aseguradoras": generar_caso7_aseguradoras
    }

    print("Generando los datasets...")

    for nombre_caso, funcion_generadora in casos.items():
        # 1. Dataset Limpio
        df_limpio, columnas_numericas = funcion_generadora()
        ruta_limpio = os.path.join(directorio_salida, f"{nombre_caso}_1_Limpio.csv")
        df_limpio.to_csv(ruta_limpio, index=False)
        
        # 2. Dataset con Errores Ligeros (5% Nulos, 2% Outliers)
        df_errores_ligeros = introducir_ruido(df_limpio, perc_nulos=0.05, perc_outliers=0.02, columnas_numericas=columnas_numericas)
        ruta_ligero = os.path.join(directorio_salida, f"{nombre_caso}_2_ErroresLigeros.csv")
        df_errores_ligeros.to_csv(ruta_ligero, index=False)
        
        # 3. Dataset con Errores Graves (15% Nulos, 10% Outliers)
        df_errores_graves = introducir_ruido(df_limpio, perc_nulos=0.15, perc_outliers=0.10, columnas_numericas=columnas_numericas)
        ruta_grave = os.path.join(directorio_salida, f"{nombre_caso}_3_ErroresGraves.csv")
        df_errores_graves.to_csv(ruta_grave, index=False)
        
        print(f"✅ Generados los 3 archivos para: {nombre_caso}")

    print(f"\n¡Proceso finalizado! Los 21 archivos CSV están disponibles en la carpeta '{directorio_salida}'.")
