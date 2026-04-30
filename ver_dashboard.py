from optuna_dashboard import run_server

# Levantamos el servidor web directamente desde el código
run_server("sqlite:///optuna_resultados_rgb_focused_v3.db", host="127.0.0.1", port=8080)