import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- CONFIGURACIÓN DE RUTAS ---
CSV_PATH = "C:/TFG/data/Original_Data/ISIC_FINAL/isic_metadata_full.csv"
OUTPUT_DIR = "C:/TFG/docs/Memoria y estado del arte/imagenes/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuración estética global
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 14, 
    'axes.titlesize': 18, 
    'axes.labelsize': 15,
    'legend.fontsize': 13
})

def generate_eda_plots():
    print("🚀 Cargando dataset y procesando etiquetas...")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # 0. Lógica de clasificación
    def classify_diagnosis(val):
        val = str(val).lower().strip()
        malignant_terms = ['melanoma', 'basal cell', 'basalioma', 'squamous cell', 'carcinoma', 'malignant']
        indeterminate_terms = ['indeterminate', 'unknown', 'unsure', 'nan', 'none']
        if any(x in val for x in malignant_terms): return 'Malignant'
        elif any(x in val for x in indeterminate_terms) or val == 'nan': return 'Indeterminate'
        else: return 'Benign'

    df['clinical_label'] = df['diagnosis_1'].apply(classify_diagnosis)
    
    # Definimos counts_check para que esté disponible en todo el script
    counts_check = df['clinical_label'].value_counts()

    # --- 1. EDAD ---
    print("📊 Gráfica 1: Edad...")
    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(12, 8))
    age_data = df['age_approx'].dropna()
    sns.boxplot(x=age_data, ax=ax_box, color="#5DADE2", fliersize=4)
    sns.histplot(age_data, kde=False, ax=ax_hist, color="#2E86C1", bins=30)
    ax_hist.set_title("Distribución de Edad y Detección de Outliers", pad=20)
    ax_hist.set_xlabel("Edad Aproximada")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_distribucion_edad.png"), dpi=300)

    # --- 2. SEXO ---
    print("📊 Gráfica 2: Sexo...")
    sex_counts = df['sex'].fillna('No definido').value_counts()
    plt.figure(figsize=(10, 8))
    wedges, _, autotexts = plt.pie(
        sex_counts, autopct='%1.1f%%', startangle=140, pctdistance=0.75,
        colors=["#3498DB", "#F1948A", "#D5DBDB"], wedgeprops=dict(width=0.55, edgecolor='w')
    )
    plt.setp(autotexts, size=16, weight="bold")
    plt.legend(wedges, sex_counts.index, title="Sexo Biológico", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title("Composición por Sexo Biológico", pad=20)
    plt.savefig(os.path.join(OUTPUT_DIR, "02_donut_sexo.png"), dpi=300, bbox_inches='tight')

    # --- 3. LOCALIZACIÓN ---
    print("📊 Gráfica 3: Localización...")
    plt.figure(figsize=(12, 8))
    order_site = df['anatom_site_general'].value_counts().index
    sns.countplot(y="anatom_site_general", data=df, order=order_site, hue="anatom_site_general", palette="viridis", legend=False)
    plt.title("Distribución por Sitio Anatómico")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_sitio_anatomico.png"), dpi=300)

    # --- 4. SESGO INSTRUMENTAL (Filtrado a Top 4) ---
    print("📊 Gráfica 4: Sesgo Instrumental (Top 4 técnicas)...")
    bias_counts_full = pd.crosstab(df['image_type'], df['clinical_label'])
    for cat in ['Benign', 'Malignant', 'Indeterminate']:
        if cat not in bias_counts_full.columns: bias_counts_full[cat] = 0
    
    # 🚩 FILTRO: Seleccionamos solo los 4 tipos con mayor volumen total
    type_order = bias_counts_full.sum(axis=1).sort_values(ascending=False).head(4).index
    bias_counts = bias_counts_full.loc[type_order]
    
    # Eje Y es porcentaje relativo (0-100%)
    bias_pct = bias_counts.div(bias_counts.sum(axis=1), axis=0) * 100
    bias_long_pct = bias_pct.reset_index().melt(id_vars='image_type', var_name='Diagnóstico', value_name='Porcentaje')
    
    plt.figure(figsize=(16, 10))
    # Definimos las categorías explícitamente para asegurar que el mapeo de n=X sea correcto
    plot_categories = ['Benign', 'Indeterminate', 'Malignant']
    
    ax = sns.barplot(data=bias_long_pct, x='image_type', y='Porcentaje', hue='Diagnóstico', 
                     palette=["#82E0AA", "#F7DC6F", "#EC7063"], order=type_order, hue_order=plot_categories)
    
    plt.title("Sesgo Instrumental: Proporción Diagnóstica en Técnicas Principales", pad=20)
    plt.ylabel("Porcentaje dentro de la técnica (%)")
    plt.xlabel("Técnica de Imagen (Top 4 por volumen de muestras)")
    plt.xticks(rotation=15, ha='center')
    plt.ylim(0, 115) 

    # Mapeo de patches para n=X
    for i, p in enumerate(ax.patches):
        h = p.get_height()
        if h > 0:
            # Porcentaje sobre la barra
            ax.text(p.get_x() + p.get_width()/2., h + 1, f'{h:.1f}%', ha="center", weight='bold', size=11)
            
            # Cantidad absoluta VERTICAL (n=X)
            # Seaborn ordena patches por 'hue' primero, luego por 'x'
            if h > 5: # Bajamos el umbral a 5% para que clinical:close-up también muestre n
                cat_idx = i // len(type_order)
                type_idx = i % len(type_order)
                cat_name = plot_categories[cat_idx]
                type_name = type_order[type_idx]
                n_val = bias_counts.loc[type_name, cat_name]
                
                ax.text(p.get_x() + p.get_width()/2., h/2, f'n={int(n_val):,}', 
                        ha="center", va="center", color="black", weight='bold', 
                        rotation=90, size=11, alpha=0.8)

    plt.legend(title="Diagnóstico")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_sesgo_instrumental.png"), dpi=300)

    # --- 5. NULOS EN CASCADA ---
    print("📊 Gráfica 5: Nulos...")
    diag_cols = ['diagnosis_1', 'diagnosis_2', 'diagnosis_3', 'diagnosis_4', 'diagnosis_5']
    null_pct = df[diag_cols].isnull().mean() * 100
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x=null_pct.index, y=null_pct.values, hue=null_pct.index, palette="mako", legend=False)
    plt.title("Frecuencia de Valores Nulos por Nivel Diagnóstico", pad=20)
    plt.ylabel("Porcentaje de Nulos (%)")
    for i, v in enumerate(null_pct.values):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', weight='bold')
    plt.ylim(0, 110)
    plt.savefig(os.path.join(OUTPUT_DIR, "05_barras_nulos.png"), dpi=300)

    # --- 6. DESBALANCEO ---
    print("📊 Gráfica 6: Desbalanceo...")
    plt.figure(figsize=(10, 7))
    target_counts = counts_check.reindex(['Benign', 'Malignant'], fill_value=0)
    ax = sns.barplot(x=target_counts.index, y=target_counts.values, hue=target_counts.index, palette=["#58D68D", "#E74C3C"], legend=False)
    plt.title("Desbalanceo entre Casos Benignos y Malignos", pad=20)
    for i, v in enumerate(target_counts.values):
        ax.text(i, v + (target_counts.max() * 0.02), f'{int(v):,}', ha='center', fontweight='bold', size=14)
    plt.savefig(os.path.join(OUTPUT_DIR, "06_desbalanceo_clase.png"), dpi=300)

    print(f"\n✅ ¡Proceso completado! Gráficas en: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_eda_plots()