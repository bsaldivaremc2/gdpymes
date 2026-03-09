import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import math
import warnings

# Ignorar advertencias menores de Seaborn sobre layouts
warnings.filterwarnings("ignore")

# ==========================================
# 1. FUNCIONES INDIVIDUALES DE GRAFICADO
# ==========================================

def plot_missing_values_cols(ax, df):
    """Muestra la cantidad de valores nulos por columna."""
    missing_cols = df.isna().sum()
    missing_cols = missing_cols[missing_cols > 0] # Solo mostrar las que tienen nulos
    if missing_cols.empty:
        ax.text(0.5, 0.5, 'Sin valores nulos en columnas', ha='center', va='center')
        ax.set_title("Nulos por Columna")
        return
    
    sns.barplot(x=missing_cols.values, y=missing_cols.index, ax=ax, palette="Reds_r")
    ax.set_title("Cantidad de Nulos por Columna")
    ax.set_xlabel("Nulos")

def plot_missing_values_rows(ax, df):
    """Muestra la distribución de filas según la cantidad de valores nulos que contienen."""
    missing_rows = df.isna().sum(axis=1)
    missing_counts = missing_rows.value_counts().sort_index()
    
    sns.barplot(x=missing_counts.index, y=missing_counts.values, ax=ax, color="salmon")
    ax.set_title("Distribución de Nulos por Fila")
    ax.set_xlabel("Cantidad de celdas nulas en la fila")
    ax.set_ylabel("Número de filas")

def plot_categorical_pie(ax, df, col):
    """Crea un gráfico de pastel para variables categóricas."""
    data = df[col].dropna().value_counts()
    if data.empty:
        ax.text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center')
        return
    
    # Limitar a top 5 categorías para evitar saturación visual
    if len(data) > 5:
        top_data = data.iloc[:4]
        top_data['Otros'] = data.iloc[4:].sum()
        data = top_data

    ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax.set_title(f"Proporción de: {col}")

def plot_numeric_dist(ax, df, col):
    """Crea un histograma con curva KDE para variables numéricas."""
    data = df[col].dropna()
    if data.empty:
        ax.text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center')
        return
    
    sns.histplot(data, kde=True, ax=ax, color="skyblue", bins=30)
    ax.set_title(f"Distribución de: {col}")
    ax.set_xlabel("")

def plot_numeric_vs_numeric(ax, df, col1, col2):
    """Crea un gráfico de dispersión (scatter) entre dos variables numéricas."""
    data = df[[col1, col2]].dropna()
    if data.empty:
        ax.text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center')
        return
    
    sns.scatterplot(data=data, x=col1, y=col2, ax=ax, alpha=0.6, color="purple")
    ax.set_title(f"Relación: {col1} vs {col2}")

def plot_numeric_vs_categorical(ax, df, num_col, cat_col):
    """Crea un boxplot para ver la distribución de un número según una categoría."""
    data = df[[num_col, cat_col]].dropna()
    if data.empty or data[cat_col].nunique() == 0:
        ax.text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center')
        return
    
    # Usar boxplot porque es más limpio que múltiples histogramas solapados
    sns.boxplot(data=data, x=cat_col, y=num_col, ax=ax, palette="Set2")
    ax.set_title(f"{num_col} agrupado por {cat_col}")
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel("")

# ==========================================
# 2. MOTOR DE GENERACIÓN DEL REPORTE
# ==========================================

def generate_report(df, output_pdf="reporte_analisis.pdf", rows=3, cols=2):
    """
    Toma un DataFrame y genera un reporte en PDF de múltiples páginas con proporción A4.
    """
    print(f"Analizando DataFrame de {df.shape[0]} filas y {df.shape[1]} columnas...")
    
    # Separar tipos de columnas (excluyendo IDs típicos que no aportan visualmente)
    cols_to_plot = [c for c in df.columns if not c.lower().startswith('id_')]
    
    num_cols = df[cols_to_plot].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[cols_to_plot].select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Construir la lista de tareas (funciones y sus argumentos)
    tasks = []
    
    # 1. Gráficos de Valores Nulos
    tasks.append((plot_missing_values_cols, {'df': df}))
    tasks.append((plot_missing_values_rows, {'df': df}))
    
    # 2. Distribuciones individuales
    for col in cat_cols:
        tasks.append((plot_categorical_pie, {'df': df, 'col': col}))
        
    for col in num_cols:
        tasks.append((plot_numeric_dist, {'df': df, 'col': col}))
        
    # 3. Relaciones (Num vs Num)
    # Hacemos pares únicos si hay más de 1 numérica
    if len(num_cols) >= 2:
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                tasks.append((plot_numeric_vs_numeric, {'df': df, 'col1': num_cols[i], 'col2': num_cols[j]}))
                
    # 4. Relaciones (Num vs Cat)
    # Tomamos la primera categórica (si existe) y la cruzamos con las numéricas para no generar cientos de gráficos
    if cat_cols and num_cols:
        main_cat = cat_cols[0] 
        for num in num_cols:
            tasks.append((plot_numeric_vs_categorical, {'df': df, 'num_col': num, 'cat_col': main_cat}))

    # Configuración de paginación A4 (8.27 x 11.69 pulgadas)
    plots_per_page = rows * cols
    total_pages = math.ceil(len(tasks) / plots_per_page)
    
    print(f"Se generarán {len(tasks)} gráficos distribuidos en {total_pages} páginas.")

    with PdfPages(output_pdf) as pdf:
        task_idx = 0
        for page in range(total_pages):
            fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69))
            fig.suptitle(f"Reporte Automático de Datos - Página {page + 1}/{total_pages}", fontsize=16, y=0.98)
            axes = axes.flatten() # Convertir matriz a array 1D para iterar fácil
            
            for ax in axes:
                if task_idx < len(tasks):
                    # Extraer función y argumentos
                    func, kwargs = tasks[task_idx]
                    func(ax=ax, **kwargs) # Ejecutar la función de graficado
                    task_idx += 1
                else:
                    # Ocultar los subplots sobrantes en la última página
                    ax.set_visible(False)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Dejar espacio para el título principal
            pdf.savefig(fig)
            plt.close(fig)
            
    print(f"✅ Reporte guardado con éxito en: {output_pdf}")

# ==========================================
# 3. EJECUCIÓN DE PRUEBA
# ==========================================
if __name__ == "__main__":
    # Puedes ejecutar este script pasándole uno de los CSVs generados anteriormente.
    import os
    
    # Simulamos leer el archivo de errores graves del Caso 1 si existe
    archivo_prueba = "CSVs/Andres_Carreteras_3_ErroresGraves.csv"
    
    if os.path.exists(archivo_prueba):
        df_prueba = pd.read_csv(archivo_prueba)
        generate_report(df_prueba, output_pdf="Reporte_Carreteras.pdf")
    else:
        print("Para probarlo, importa este script en tu notebook y usa la función: generate_report(tu_dataframe)")
