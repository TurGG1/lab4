import subprocess
import os

if __name__ == "__main__":
    # Читаем путь к лучшей модели
    with open("best_model.txt", "r") as f:
        model_path = f.read().strip()
    
    print(f"Deploying model from: {model_path}")
    
    # Запускаем MLflow serve на порту 5003
    cmd = f"mlflow models serve -m {model_path} -p 5003 --no-conda &"
    subprocess.Popen(cmd, shell=True)
    print("Model deployed on port 5003")


