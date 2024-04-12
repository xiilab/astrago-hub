import torch
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings 
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.metrics import mean_squared_error

def load_data(data_dir):
    parent_dir = os.path.abspath(data_dir)
    # train dataset
    train_path = os.path.join(parent_dir, 'train')
    # test dataset
    test_path = os.path.join(parent_dir, 'test')

    train_path = [os.path.join(train_path, file) for file in os.listdir(train_path) if file.endswith('.csv')]
    test_path = [os.path.join(test_path, file) for file in os.listdir(test_path) if file.endswith('.csv')]

    return train_path, test_path

def preprocess(train_path, test_path):
    df = pd.DataFrame()
    for file_path in train_path:
        df_each = pd.read_csv(file_path)
        df = pd.concat([df, df_each], ignore_index=True)

    df = df[df['data_num'] != 0]
    df = df.reset_index()
    df.drop('index', axis=1, inplace=True)

    X = df.drop('total_data_inference_time', axis=1).values
    y = df['total_data_inference_time'].values

    # data slice
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

    # torch tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    

    df = pd.DataFrame()
    for file_path in test_path:
        df_each = pd.read_csv(file_path)
        df_each['total_inference_time'] = df_each['single_data_inference_time'].cumsum()
        df = pd.concat([df, df_each], ignore_index=True)

    df_test = df[df['inference_time'] != 0]  # 0 epoch 제거
    df_test.drop("model_name", axis=1, inplace=True)
    df_test.drop("data_num", axis=1, inplace=True)
    df_test.drop("gpu_usage", axis=1, inplace=True)
    df_test.drop("cpu_usage", axis=1, inplace=True)
    df_test.drop("inference_time", axis=1, inplace=True)
    df_test.drop("save_time", axis=1, inplace=True)
    df_test.drop('single_data_inference_time', axis=1, inplace=True)
    df_test.drop('gpu', axis=1, inplace=True)
    df_test.rename(columns={'num': 'data_num'}, inplace=True)
    df_test = df_test[['FLOPS', 'data_num', 'imgsz', 'param', 'total_inference_time']]

    df_test['FLOPS'] = 14

    X_test = df_test.drop('total_inference_time', axis=1).values
    y_test = df_test['total_inference_time'].values

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test



def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, n_estimator, max_depth, learning_rate):
    # XGBoost 회귀 모델 정의
    model = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=learning_rate, min_child_weight=5, subsample=0.45,
                              max_depth=max_depth, n_estimators=n_estimator, tree_method='gpu_hist', gpu_id=0)

    # 모델 학습
    model.fit(X_train, y_train)

    # 훈련 데이터셋에 대한 성능 평가
    train_preds = model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_preds)
    print(f'Train MSE: {train_mse:.4f}')

    # 검증 데이터셋에 대한 성능 평가
    val_preds = model.predict(X_val)  # X_val: 검증 데이터셋
    val_mse = mean_squared_error(y_val, val_preds)  # y_val: 검증 데이터의 실제 값
    print(f'Validation MSE: {val_mse:.4f}')

    # 테스트 데이터셋에 대한 성능 평가
    y_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    print(f'Test MSE: {test_mse:.4f}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--data", default='data/', type=str, help="Directory containing train and test data")
    parser.add_argument("--ne", default=10000, type=int, help="n_estimator")
    parser.add_argument("--md", default=9, type=int, help="max_depth")
    parser.add_argument("--lr", default=0.05, type=float, help="learning rate")
    args = parser.parse_args()

    data_dir = args.data
    n_estimator = args.ne
    max_depth = args.md
    learning_rate = args.lr
    
    train_dir, test_dir = load_data(data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess(train_dir, test_dir)
    train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, n_estimator, max_depth, learning_rate)
