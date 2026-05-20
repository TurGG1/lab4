import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import os

def download_data():
    
    df = pd.read_csv('bmw_global_sales_2018_2025.csv')
    
    
    df.to_csv("bmw_raw.csv", index=False)
    print(f"Data downloaded. Shape: {df.shape}")
    return df

def preprocess_data(path2df):
    df = pd.read_csv(path2df)
    
    # Категориальные колонки
    cat_columns = ['Region', 'Model']
    
    # Очистка данных от аномалий
    question_revenue = df[df['Revenue_EUR'] <= 0]
    df = df.drop(question_revenue.index)
    
    question_units = df[df['Units_Sold'] <= 0]
    df = df.drop(question_units.index)
    
    question_price_low = df[df['Avg_Price_EUR'] < 10000]
    df = df.drop(question_price_low.index)
    
    question_price_high = df[df['Avg_Price_EUR'] > 200000]
    df = df.drop(question_price_high.index)
    
    question_bev = df[(df['BEV_Share'] < 0) | (df['BEV_Share'] > 1)]
    df = df.drop(question_bev.index)
    
    question_year = df[(df['Year'] < 2018) | (df['Year'] > 2025)]
    df = df.drop(question_year.index)
    
    df = df.reset_index(drop=True)
    
    # Кодирование категориальных признаков
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    
    df.to_csv('bmw_clear.csv', index=False)
    print(f"Preprocessed data shape: {df.shape}")
    return True

if __name__ == "__main__":
    download_data()
    preprocess_data("bmw_raw.csv")


