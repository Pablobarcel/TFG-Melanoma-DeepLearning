# src/training/train_rgb/tune_hyperparams.py

from src.training.train_rgb import train_model

def main():
    print("=============================================")
    print("🔎 TUNING: LR + MOMENTUM (Con Early Stopping)")
    print("=============================================")

    # 1. Hiperparámetros a probar
    # LR: Probamos uno "lento" y uno "rápido"
    learning_rates = [1e-3, 1e-4]
    
    # Momentum: 0.9 es estándar, 0.99 es para "acelerar" más
    momentums = [0.9, 0.99]
    
    # Configuración fija
    BATCH_SIZE = 32
    EPOCHS_MAX = 10  # Ponemos margen, el Early Stopping cortará antes si hace falta
    PATIENCE = 3     # Si en 3 epochs no mejora la Loss, paramos
    
    total_runs = len(learning_rates) * len(momentums)
    current_run = 1

    for lr in learning_rates:
        for mom in momentums:
                
            print(f"\n>>> EXPERIMENTO {current_run}/{total_runs}")
            print(f"    LR: {lr} | Momentum: {mom}")
            
            try:
                train_model(
                    learning_rate=lr,
                    momentum=mom,
                    batch_size=BATCH_SIZE,
                    patience=PATIENCE,
                    num_epochs=EPOCHS_MAX,
                    experiment_name="tuning_momentum"
                )
            except Exception as e:
                print(f"❌ Error en este run: {e}")
            
            current_run += 1

    print("\n\n🎉 TUNING FINALIZADO.")
    print("Ejecuta: python -m tensorboard --logdir experiments/tuning_momentum")

if __name__ == "__main__":
    main()