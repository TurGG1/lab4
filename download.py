import pandas as pd
import os

def download():
    # Загружаем данные из интернета или из локального файла
    url = 'https://raw.githubusercontent.com/dayekb/Basic_ML_Alg/main/cars_moldova_no_dup.csv'
    df = pd.read_csv(url)
    
    # Сохраняем
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/raw_data.csv', index=False)
    print(f"Data downloaded. Shape: {df.shape}")
    return df.shape

if __name__ == "__main__":
    download()


