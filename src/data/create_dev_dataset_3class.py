import pandas as pd
from pathlib import Path
from src.config.paths import SPLITTED_DATA_DIR

RANDOM_STATE = 42

# Objetivos ACTUALIZADOS (Clases 2 y 3 fusionadas en NMSC)
TOTAL_TARGETS = {
    0: 300000,  # BEN
    1: 10000,    # MEL
    2: 10000     # NMSC (BCC 6964 + SCC+ 2013)
    # La clase 3 ya no existe
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

    # ---> 🚨 FUSIÓN DE CLASES (NMSC) <---
    # Convertimos todos los SCC+ (3) en BCC (2) para formar la clase NMSC
    df.loc[df["head_B_label"] == 3, "head_B_label"] = 2
    print("✅ Fusión completada: Clase 3 (SCC+) convertida a Clase 2 (NMSC).")

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
    out_dir = SPLITTED_DATA_DIR / "Final_dataset_3class_300k"
    out_dir.mkdir(exist_ok=True)

    df_train.to_csv(out_dir / "train.csv", index=False)
    df_val.to_csv(out_dir / "val.csv", index=False)

    print(f"\n✅ CSVs generados correctamente y guardados en {out_dir}")


if __name__ == "__main__":
    main()