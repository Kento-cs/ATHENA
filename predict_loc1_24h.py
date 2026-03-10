import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("データを読み込み中...")
# ETTh1データの読み込み
df = pd.read_csv("data/ETTh1.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# --- 1. 特徴量エンジニアリング ---
target_col = 'OT'
features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
horizon = 24  

df['target'] = df[target_col].shift(-horizon)

lag_hours = 24
for col in features:
    # ① ラグ
    for i in range(1, lag_hours + 1):
        df[f'{col}_lag_{i}'] = df[col].shift(i)
        
    # ② 変化量
    df[f'{col}_diff1'] = df[col].diff(1)
    df[f'{col}_diff24'] = df[col].diff(24)
    
    # ③ 加速度
    df[f'{col}_accel'] = df[f'{col}_diff1'].diff(1)
    
    # ④ 移動平均
    df[f'{col}_rolling_mean24'] = df[col].rolling(window=24).mean()
    df[f'{col}_rolling_std24'] = df[col].rolling(window=24).std()

df['hour'] = df.index.hour
df['month'] = df.index.month
df.dropna(inplace=True)

# --- 2. データの分割 ---
total_len = len(df)
train_end = int(total_len * 0.7)
val_end = int(total_len * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

drop_cols = ['target', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
X_train, y_train = train_df.drop(columns=drop_cols), train_df['target']
X_val, y_val = val_df.drop(columns=drop_cols), val_df['target']
X_test, y_test = test_df.drop(columns=drop_cols), test_df['target']

train_weights = np.abs(train_df['OT'].diff(1).fillna(0)) + 1.0

# --- 3. LightGBMモデルの学習 ---
model = lgb.LGBMRegressor(n_estimators=1000, random_state=42, learning_rate=0.05)
model.fit(
    X_train, y_train,
    sample_weight=train_weights, 
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

# --- 4. テストデータでの評価と可視化 ---
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

plt.figure(figsize=(14, 5))
plt.plot(y_test.index[:500], y_test.values[:500], label='Actual (OT)', color='black', linewidth=1.5, alpha=0.7)
plt.plot(y_test.index[:500], y_pred[:500], label='Predicted (Improved 24h)', color='red', linestyle='--', linewidth=1.5)

plt.title(f'Location 1 (ETTh1): Prediction vs Actual (24h Horizon)\nOverall Test MAE: {mae:.2f}', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Oil Temperature')
plt.legend()
plt.tight_layout()
plt.savefig('spike_detection_ETTh1.png')
print("'spike_detection_ETTh1' として保存しました。")