from optuna_dashboard import run_server

# Levantamos el servidor web directamente desde el código
run_server("sqlite:///optuna_resultados_rgb.db", host="127.0.0.1", port=8080)