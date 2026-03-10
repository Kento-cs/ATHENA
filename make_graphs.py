import pandas as pd
import matplotlib.pyplot as plt
import os

data_dir = "data/" 
datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']

plt.style.use('seaborn-v0_8-darkgrid')

# それぞれのデータを読み込み、グラフ化
for name in datasets:
    file_path = os.path.join(data_dir, f"{name}.csv")
    
    # ファイルが存在するか確認
    if not os.path.exists(file_path):
        print(f" エラー: {file_path} が見つかりません。パスを確認してください。")
        continue
        
    # データの読み込みと日時インデックスの設定
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # オイル温度（OT）の可視化
    plt.figure(figsize=(14, 4))
    plt.plot(df.index, df['OT'], label=f'{name} - Oil Temperature', color='tab:blue', linewidth=0.5)
    
    # グラフの装飾
    plt.title(f'Overall Time Series of Oil Temperature (OT) - {name}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Temperature', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    # グラフを画像として保存し、画面にも表示
    output_filename = f'{name}_ot_timeseries.png'
    plt.savefig(output_filename)
    
    print(f" {name} のグラフを {output_filename} として保存しました。\n")