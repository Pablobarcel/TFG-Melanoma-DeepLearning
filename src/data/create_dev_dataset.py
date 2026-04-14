import pandas as pd
from pathlib import Path
from src.config.paths import SPLITTED_DATA_DIR

RANDOM_STATE = 42

# Objetivos ORIGINALES (4 Clases: BEN, MEL, BCC, SCC+)
TOTAL_TARGETS = {
    0: 200000,  # BEN
    1: 10000,   # MEL
    2: 6964,    # BCC
    3: 2013     # SCC+
}

TRAIN_RATIO = 0.8


def main():

    # --------------------------------------------------
    # 1. Cargar dataset grande
    # --------------------------------------------------
    df = pd.read_csv(SPLITTED_DATA_DIR / "training_with_paths.csv")

    required_cols = [
        "master_group_id",
        "head_A_label",
        "head_B_label"
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Falta columna: {c}")

    print("✅ Dataset cargado. Manteniendo 4 clases originales (Sin fusión NMSC).")

    # --------------------------------------------------
    # 2. Tabla a nivel de grupo
    # --------------------------------------------------
    group_df = (
        df.groupby("master_group_id")
        .agg(
            head_A_label=("head_A_label", "first"),
            head_B_label=("head_B_label", "first"),
            n_images=("master_group_id", "size")
        )
        .reset_index()
    )

    # --------------------------------------------------
    # 3. Seleccionar grupos por clase Head B
    # --------------------------------------------------
    selected_group_ids = []

    for cls, target_imgs in TOTAL_TARGETS.items():
        groups_cls = group_df[group_df["head_B_label"] == cls]
        groups_cls = groups_cls.sample(frac=1, random_state=RANDOM_STATE)

        acc = 0
        for _, row in groups_cls.iterrows():
            selected_group_ids.append(row["master_group_id"])
            acc += row["n_images"]
            if acc >= target_imgs:
                break

        print(f"Clase {cls}: ~{acc} imágenes seleccionadas.")

    # --------------------------------------------------
    # 4. Construir subdataset base
    # --------------------------------------------------
    df_base = df[df["master_group_id"].isin(selected_group_ids)].copy()
    df_base = df_base.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\nSubdataset base (Distribución Final):")
    print(df_base["head_A_label"].value_counts(normalize=True))
    print(df_base["head_B_label"].value_counts(normalize=True))

    # --------------------------------------------------
    # 5. Split train / val por grupos y por clase
    # --------------------------------------------------
    train_groups = []
    val_groups = []

    for cls in TOTAL_TARGETS.keys():
        groups_cls = group_df[
            (group_df["master_group_id"].isin(selected_group_ids)) &
            (group_df["head_B_label"] == cls)
        ]

        groups_cls = groups_cls.sample(frac=1, random_state=RANDOM_STATE)

        n_train = int(len(groups_cls) * TRAIN_RATIO)
        train_groups.extend(groups_cls.iloc[:n_train]["master_group_id"])
        val_groups.extend(groups_cls.iloc[n_train:]["master_group_id"])

    # --------------------------------------------------
    # 6. Crear DataFrames finales
    # --------------------------------------------------
    df_train = df_base[df_base["master_group_id"].isin(train_groups)]
    df_val = df_base[df_base["master_group_id"].isin(val_groups)]

    # --------------------------------------------------
    # 7. Comprobaciones finales
    # --------------------------------------------------
    print("\nTRAIN (Distribución):")
    print(df_train["head_A_label"].value_counts(normalize=True))
    print(df_train["head_B_label"].value_counts(normalize=True))

    print("\nVAL (Distribución):")
    print(df_val["head_A_label"].value_counts(normalize=True))
    print(df_val["head_B_label"].value_counts(normalize=True))

    # --------------------------------------------------
    # 8. Guardar CSVs
    # --------------------------------------------------
    # Cambiamos el nombre de la carpeta de salida para no sobreescribir los de 3 clases
    out_dir = SPLITTED_DATA_DIR / "Final_dataset_4class_200k" 
    out_dir.mkdir(exist_ok=True)

    df_train.to_csv(out_dir / "train.csv", index=False)
    df_val.to_csv(out_dir / "val.csv", index=False)

    print(f"\n✅ CSVs de 4 Clases generados correctamente y guardados en {out_dir}")

if __name__ == "__main__":
    main()