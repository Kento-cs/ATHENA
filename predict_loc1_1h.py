import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("データを読み込み中...")
df = pd.read_csv("data/ETTm1.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# --- 1. 特徴量エンジニアリング ---
target_col = 'OT'
features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
horizon = 4 

df['target'] = df[target_col].shift(-horizon)

for col in features:
    # ① ラグ特徴量
    for i in range(1, 13):
        df[f'{col}_lag_{i}'] = df[col].shift(i)
    
    # ② 変化量
    df[f'{col}_diff1'] = df[col].diff(1)
    df[f'{col}_diff4'] = df[col].diff(4)
    
    # ③ 加速度
    df[f'{col}_accel'] = df[f'{col}_diff1'].diff(1)
    
    # ④ 移動平均と標準偏差
    df[f'{col}_rolling_mean4'] = df[col].rolling(window=4).mean()
    df[f'{col}_rolling_std4'] = df[col].rolling(window=4).std()

df['hour'] = df.index.hour
df.dropna(inplace=True)

# --- 2. データ分割 ---
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

# --- 学習重み設定 ---
train_weights = np.abs(train_df['OT'].diff(1).fillna(0)) + 1.0

# --- 3. モデルの学習 ---
model = lgb.LGBMRegressor(n_estimators=1000, random_state=42, learning_rate=0.05)
model.fit(
    X_train, y_train,
    sample_weight=train_weights, 
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

# --- 4. 評価と可視化 ---
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# グラフ①：1ヶ月間の全体表示
plot_mask_1m = (y_test.index >= '2018-05-01') & (y_test.index < '2018-06-01')
plt.figure(figsize=(14, 5))

plt.plot(y_test[plot_mask_1m].index, y_test[plot_mask_1m].values, label='Actual (OT)', color='black', linewidth=1.5, alpha=0.7)
plt.plot(y_test[plot_mask_1m].index, y_pred[y_test.index.isin(y_test[plot_mask_1m].index)], label='Predicted (Lag-Corrected)', color='red', linestyle='--', linewidth=1.0)

plt.title(f'Lag-Corrected Prediction (Location: ETTm1)\nPeriod: May 2018 | Overall Test MAE: {mae:.2f}', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Oil Temperature')
plt.legend()
plt.tight_layout()

plt.savefig('spike_detection_ETTm1_1month.png')
print("1ヶ月のグラフを 'spike_detection_ETTm1_1month.png' に保存しました。")

# グラフ②：最初の3日間にズーム（★追加部分）
plot_mask_3d = (y_test.index >= '2018-05-01') & (y_test.index < '2018-05-04')
plt.figure(figsize=(14, 5))

# ズーム版は細かい点（データポイント）が見えるようにマーカー（marker='x'等）をつけるのがおすすめ
plt.plot(y_test[plot_mask_3d].index, y_test[plot_mask_3d].values, label='Actual (OT)', color='black', linewidth=2.0)
plt.plot(y_test[plot_mask_3d].index, y_pred[y_test.index.isin(y_test[plot_mask_3d].index)], label='Predicted (Lag-Corrected)', color='red', linestyle='--', linewidth=1.5, marker='x', markersize=4)

plt.title(f'Zoomed View: First 3 Days of May (Location: ETTm1)\nChecking for Peak Tracking and Lag', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Oil Temperature')
plt.legend()
plt.tight_layout()

plt.savefig('spike_detection_ETTm1_3days.png')
print("3日間のズームグラフを 'spike_detection_ETTm1_3days.png' に保存しました。")