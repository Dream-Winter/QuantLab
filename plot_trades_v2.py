"""
Visualize trading behavior for v2 (simple) model
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from env_v2 import RealisticHKEnv

def main():
    print("Loading v2_final model (refactored rewards) and generating visualization...")

    # Load model
    model_path = 'ppo_hkstock_v2_final.zip'
    try:
        model = PPO.load(model_path)
        print(f"✓ Loaded model from {model_path}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("Please run train_v2_simple.py first to train the model.")
        return

    # Create environment (with features, matching training config)
    env = RealisticHKEnv(
        data_path='data/00700_hist.csv',
        benchmark_path=None,  # 不使用基准（训练时也没有）
        use_features=True,    # 使用特征（训练时启用了）
        enable_sharpe_penalty=False,  # 测试时关闭以简化
        feature_window=30     # 训练时使用30天窗口
    )

    # Run full test episode
    obs, info = env.reset()
    prices = []
    actions = []
    total_assets = []
    dates = []
    rewards = []

    done = False
    step = 0

    while not done and step < len(env.df) - 5:
        action, _states = model.predict(obs, deterministic=True)
        actions.append(int(action))

        # Record before step
        current_idx = env.current_step
        prices.append(env.df.iloc[current_idx]['close'])
        dates.append(env.df.iloc[current_idx]['date'])
        total_assets.append(info['total_asset'])

        # Execute step
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        step += 1

    print(f"Episode completed: {step} steps")
    print(f"Total trading actions (non-hold): {sum(1 for a in actions if a != 0)}")
    print(f"Final total asset: {info['total_asset']:.2f} HKD")

    # Count trades
    buys = sum(1 for a in actions if a == 1)
    sells = sum(1 for a in actions if a == 2)
    print(f"Buy signals: {buys}, Sell signals: {sells}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(hspace=0.05)

    # Plot 1: Price with trade markers
    ax1.plot(dates, prices, 'k-', linewidth=1.5, alpha=0.8, label='Close Price')
    ax1.set_ylabel('Price (HKD)', fontsize=12)
    ax1.set_title('Trading Behavior - v2_simple (Realistic Costs)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Mark buy points
    buy_indices = [i for i, a in enumerate(actions) if a == 1]
    if buy_indices:
        buy_dates = [dates[i] for i in buy_indices]
        buy_prices = [prices[i] for i in buy_indices]
        ax1.scatter(buy_dates, buy_prices, color='red', s=80, marker='^',
                   label='BUY', zorder=5, edgecolors='black', linewidth=1)

    # Mark sell points
    sell_indices = [i for i, a in enumerate(actions) if a == 2]
    if sell_indices:
        sell_dates = [dates[i] for i in sell_indices]
        sell_prices = [prices[i] for i in sell_indices]
        ax1.scatter(sell_dates, sell_prices, color='green', s=80, marker='v',
                   label='SELL', zorder=5, edgecolors='black', linewidth=1)

    ax1.legend(loc='upper left')

    # Plot 2: Total asset curve
    ax2.plot(dates, total_assets, 'b-', linewidth=1.5, label='Total Asset')
    ax2.fill_between(dates, total_assets, alpha=0.3, color='blue')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Asset (HKD)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    # Add horizontal line at initial balance
    ax2.axhline(y=env.initial_balance, color='red', linestyle='--',
                alpha=0.5, label='Initial (20,000 HKD)')
    ax2.legend(loc='upper left')

    # Final stats
    initial = env.initial_balance
    final = total_assets[-1] if total_assets else initial
    profit = final - initial
    profit_pct = (profit / initial) * 100
    ax2.set_title(f'Final: {final:.2f} HKD ({profit:+.2f}, {profit_pct:+.2f}%)',
                  fontsize=12)

    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = 'data/trading_behavior_v2_simple.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    # Additional analysis
    print("\n" + "="*60)
    print("BEHAVIOR ANALYSIS")
    print("="*60)

    # Action distribution
    action_counts = pd.Series(actions).value_counts().sort_index()
    print("\nAction distribution:")
    for act in [0, 1, 2]:
        count = action_counts.get(act, 0)
        pct = (count / len(actions)) * 100
        act_name = {0: 'Hold', 1: 'Buy', 2: 'Sell'}[act]
        print(f"  {act_name}: {count} ({pct:.1f}%)")

    # Consecutive actions
    consecutive_holds = 0
    max_consecutive_holds = 0
    for a in actions:
        if a == 0:
            consecutive_holds += 1
            max_consecutive_holds = max(max_consecutive_holds, consecutive_holds)
        else:
            consecutive_holds = 0

    print(f"\nMax consecutive holds: {max_consecutive_holds}")
    print(f"Total profit/loss: {profit:.2f} HKD ({profit_pct:.2f}%)")
    print("="*60)

    plt.close()

if __name__ == "__main__":
    main()