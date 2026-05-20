import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib

def scale_frame(frame):
    df = frame.copy()
    feature_columns = ['Year', 'Month', 'Units_Sold', 'Avg_Price_EUR', 
                       'BEV_Share', 'Premium_Share', 'GDP_Growth', 'Fuel_Price_Index',
                       'Region', 'Model']
    X = df[feature_columns]
    y = df['Revenue_EUR']
    
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1, 1))
    return X_scale, Y_scale, power_trans, scaler, feature_columns

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    df = pd.read_csv("bmw_clear.csv")
    
    X, Y, power_trans, scaler, feature_columns = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    mlflow.set_experiment("bmw_revenue_prediction")
    
    with mlflow.start_run():
        rf = RandomForestRegressor(random_state=42, n_jobs=4)
        clf = GridSearchCV(rf, params, cv=3, n_jobs=4, scoring='r2')
        clf.fit(X_train, y_train.reshape(-1))
        
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        
        y_val_real = power_trans.inverse_transform(y_val)
        y_pred_real = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        
        rmse, mae, r2 = eval_metrics(y_val_real, y_pred_real)
        
        mlflow.log_params({
            "n_estimators": best.n_estimators,
            "max_depth": best.max_depth,
            "min_samples_split": best.min_samples_split
        })
        
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        
        # Сохраняем модель локально
        joblib.dump(best, "bmw_model.pkl")
        
        # Логируем в MLflow
        signature = infer_signature(X_train, best.predict(X_train))
        mlflow.sklearn.log_model(best, "model", signature=signature)
        
        print(f"RMSE: {rmse:.0f}, MAE: {mae:.0f}, R2: {r2:.4f}")
    
    # Получаем путь к лучшей модели для deploy
    dfruns = mlflow.search_runs()
    path2model = dfruns.sort_values("metrics.r2", ascending=False).iloc[0]['artifact_uri'].replace("file://", "") + '/model'
    print(path2model)
    
    # Записываем путь в файл для Jenkins
    with open("best_model.txt", "w") as f:
        f.write(path2model)


