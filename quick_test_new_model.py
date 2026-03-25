"""
快速测试新模型：检查是否产生交易
"""
import os
import numpy as np
from stable_baselines3 import PPO
from env_v2 import RealisticHKEnv

# 设置PYTHONPATH
os.environ['PYTHONPATH'] = '/app'

def test_model():
    print("="*60)
    print("TESTING NEW V2 MODEL (Refactored Rewards)")
    print("="*60)

    # 加载新训练的模型
    try:
        model = PPO.load('ppo_hkstock_v2_final.zip')
        print("✓ Loaded new v2 model")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    # 创建环境
    env = RealisticHKEnv(
        data_path='data/00700_hist.csv',
        benchmark_path=None,  # 无需基准
        use_features=True,    # 使用特征
        enable_sharpe_penalty=False,  # 测试时关闭Sharpe惩罚以便观察
        feature_window=30
    )

    # 运行一个完整episode
    obs, info = env.reset()
    done = False
    step = 0

    actions = []
    positions = []
    total_assets = []
    prices = []
    rewards = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        action_int = int(action)

        actions.append(action_int)
        positions.append(env.position)
        total_assets.append(info['total_asset'])
        prices.append(env.df.iloc[env.current_step-1]['close'])

        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        step += 1

    # 统计交易
    trades = 0
    for i in range(1, len(actions)):
        if actions[i] == 1 and actions[i-1] != 1:  # Buy
            trades += 1
        elif actions[i] == 2 and actions[i-1] != 2:  # Sell
            trades += 1

    print(f"\nEpisode: {step} steps")
    print(f"Initial Asset: {total_assets[0]:.2f} HKD")
    print(f"Final Asset: {total_assets[-1]:.2f} HKD")
    profit = total_assets[-1] - total_assets[0]
    profit_pct = (profit / total_assets[0]) * 100
    print(f"Profit: {profit:.2f} HKD ({profit_pct:.2f}%)")
    print(f"\n🎯 Total Trades: {trades}")
    print(f"  Buy actions: {sum(1 for a in actions if a == 1)}")
    print(f"  Sell actions: {sum(1 for a in actions if a == 2)}")
    print(f"  Hold actions: {sum(1 for a in actions if a == 0)}")

    # 检查是否至少有一笔交易
    if trades > 0:
        print("\n✅ SUCCESS: Model is trading!")
    else:
        print("\n❌ Model still not trading (Total Trades = 0)")

    # 打印奖励统计
    print(f"\nReward stats:")
    print(f"  Mean: {np.mean(rewards):.6f}")
    print(f"  Max: {np.max(rewards):.6f}")
    print(f"  Min: {np.min(rewards):.6f}")
    print(f"  Std: {np.std(rewards):.6f}")

    # 检查奖励是否在-0.1到0.1范围内
    max_abs_reward = np.max(np.abs(rewards))
    print(f"\nReward range check: max |reward| = {max_abs_reward:.6f}")
    if max_abs_reward <= 0.1:
        print("✅ Rewards normalized within [-0.1, 0.1]")
    else:
        print("⚠️  Some rewards exceed 0.1 threshold")

if __name__ == "__main__":
    test_model()
