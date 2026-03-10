import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("比較グラフを作成")

#  1. 特徴量エンジニアリング
def prepare_data(file_path, location_id):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    target_col = 'OT'
    features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    horizon = 4 # 1時間後を予測
    
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
        
        # ④ 移動平均
        df[f'{col}_rolling_mean4'] = df[col].rolling(window=4).mean()
        df[f'{col}_rolling_std4'] = df[col].rolling(window=4).std()

    df['hour'] = df.index.hour
    df['location'] = location_id 
    df.dropna(inplace=True)
    return df

def split_data(df):
    total_len = len(df)
    train_end = int(total_len * 0.7)
    val_end = int(total_len * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

# データの読み込み
df_m1 = prepare_data("data/ETTm1.csv", location_id=1)
df_m2 = prepare_data("data/ETTm2.csv", location_id=2)

train_m1, val_m1, test_m1 = split_data(df_m1)
train_m2, val_m2, test_m2 = split_data(df_m2)
train_mix = pd.concat([train_m1, train_m2])
val_mix = pd.concat([val_m1, val_m2])

drop_cols = ['target', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
X_train_m1, y_train_m1 = train_m1.drop(columns=drop_cols), train_m1['target']
X_val_m1, y_val_m1 = val_m1.drop(columns=drop_cols), val_m1['target']
X_test_m1, y_test_m1 = test_m1.drop(columns=drop_cols), test_m1['target']

X_train_mix, y_train_mix = train_mix.drop(columns=drop_cols), train_mix['target']
X_val_mix, y_val_mix = val_mix.drop(columns=drop_cols), val_mix['target']

# --- 2. 設定 ---
weight_m1 = np.abs(train_m1['OT'].diff(1).fillna(0)) + 1.0
weight_mix = np.abs(train_mix['OT'].diff(1).fillna(0)) + 1.0

# --- 3. 学習 ---
model_s = lgb.LGBMRegressor(n_estimators=1000, random_state=42, learning_rate=0.05)
model_s.fit(X_train_m1, y_train_m1, 
            sample_weight=weight_m1, 
            eval_set=[(X_val_m1, y_val_m1)], 
            callbacks=[lgb.early_stopping(50, verbose=False)])

model_m = lgb.LGBMRegressor(n_estimators=1000, random_state=42, learning_rate=0.05)
model_m.fit(X_train_mix, y_train_mix, 
            sample_weight=weight_mix, 
            eval_set=[(X_val_mix, y_val_mix)], 
            categorical_feature=['location'],
            callbacks=[lgb.early_stopping(50, verbose=False)])

# --- 4. 評価と予測 ---
pred_s = model_s.predict(X_test_m1)
pred_m = model_m.predict(X_test_m1)

mae_s_full = mean_absolute_error(y_test_m1, pred_s)
mae_m_full = mean_absolute_error(y_test_m1, pred_m)

# 描画期間の設定（2018年5月）
plot_mask = (y_test_m1.index >= '2018-05-01') & (y_test_m1.index < '2018-06-01')
y_test_p, pred_s_p, pred_m_p = y_test_m1[plot_mask], pred_s[plot_mask], pred_m[plot_mask]
zoom_steps = 288 

# ==========================================
# グラフ：予測 vs 実測
# ==========================================
fig1, (ax1_1, ax1_2) = plt.subplots(2, 1, figsize=(15, 10))
# 1ヶ月表示
ax1_1.plot(y_test_p.index, y_test_p.values, label='Actual', color='black', alpha=0.6)
ax1_1.plot(y_test_p.index, pred_s_p, label=f'Specialized (MAE: {mae_s_full:.3f})', color='blue', linestyle='--')
ax1_1.plot(y_test_p.index, pred_m_p, label=f'Mixed (MAE: {mae_m_full:.3f})', color='red', linestyle='-.')
ax1_1.set_title('Prediction Comparison: 1 Month (Lag-Corrected version)')
ax1_1.legend()

# 3日間表示
ax1_2.plot(y_test_p.index[:zoom_steps], y_test_p.values[:zoom_steps], label='Actual', color='black', linewidth=2)
ax1_2.plot(y_test_p.index[:zoom_steps], pred_s_p[:zoom_steps], label='Specialized', color='blue', marker='o', markersize=3, linestyle='--')
ax1_2.plot(y_test_p.index[:zoom_steps], pred_m_p[:zoom_steps], label='Mixed', color='red', marker='x', markersize=3, linestyle='-.')
ax1_2.set_title('Zoomed View: First 3 Days of May')
ax1_2.legend()
plt.tight_layout()
plt.savefig('prediction_comparison.png')

print("グラフを 'prediction_comparison.png' として保存しました。")