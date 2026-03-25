"""
按时间切分数据集：训练集（~2025-08）和测试集（2025-09~）
"""
import pandas as pd
from datetime import datetime

def split_data():
    # 加载原始数据
    df = pd.read_csv('data/00700_hist.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 定义切分点
    split_date = datetime(2025, 8, 31)  # 8月31日及之前为训练集

    # 切分
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()

    print("="*60)
    print("DATASET SPLIT SUMMARY")
    print("="*60)
    print(f"Original data: {len(df)} rows")
    print(f"  Date range: {df['date'].iloc[0].strftime('%Y-%m-%d')} to {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print()
    print(f"Train set (until 2025-08-31): {len(train_df)} rows")
    print(f"  Range: {train_df['date'].iloc[0].strftime('%Y-%m-%d')} to {train_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print()
    print(f"Test set (2025-09-01 onwards): {len(test_df)} rows")
    print(f"  Range: {test_df['date'].iloc[0].strftime('%Y-%m-%d')} to {test_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print("="*60)

    # 保存
    train_df.to_csv('data/00700_train.csv', index=False)
    test_df.to_csv('data/00700_test.csv', index=False)

    print(f"\n✓ Saved train set to: data/00700_train.csv")
    print(f"✓ Saved test set to: data/00700_test.csv")

if __name__ == "__main__":
    split_data()
