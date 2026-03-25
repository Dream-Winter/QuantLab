"""
Test v1 model (no transaction costs) on recent 200-day sideways period.
This serves as a baseline: can the agent profit even without friction?
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from env import HKStockEnv  # Original v1 env

def main():
    print("="*60)
    print("V1 SIDEWAYS PERIOD TEST (No Transaction Costs)")
    print("="*60)

    # Load full data and identify recent 200 days
    df_full = pd.read_csv('data/00700_hist.csv')
    df_full['date'] = pd.to_datetime(df_full['date'])
    df_full = df_full.sort_values('date').reset_index(drop=True)

    # Use only last 200 days (sideways period)
    df_sideways = df_full.tail(200).reset_index(drop=True)
    print(f"Testing on recent {len(df_sideways)} days (sideways period)")
    print(f"Date range: {df_sideways['date'].iloc[0]} to {df_sideways['date'].iloc[-1]}")
    print(f"Price change: {df_sideways['close'].iloc[0]:.2f} -> {df_sideways['close'].iloc[-1]:.2f}")
    print(f"Return: {(df_sideways['close'].iloc[-1]/df_sideways['close'].iloc[0]-1)*100:.2f}%")

    # Save sideways data temporarily
    df_sideways.to_csv('data/00700_hist_sideways.csv', index=False)

    # Create environment with sideways data
    env = HKStockEnv(data_path='data/00700_hist_sideways.csv')

    # Load v1 model
    try:
        model = PPO.load('ppo_hkstock_final.zip')
        print("✓ Loaded v1 model (trained with 30-day data)")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    # Run full episode
    obs, info = env.reset()
    prices = []
    actions = []
    total_assets = []
    dates = []
    rewards = []

    done = False
    step = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        actions.append(int(action))

        current_idx = env.current_step
        prices.append(env.df.iloc[current_idx]['close'])
        dates.append(env.df.iloc[current_idx]['date'])
        # v1 env uses 'total_asset' in info, but double-check
        total_assets.append(info.get('total_asset', env._get_total_asset(env.df.iloc[current_idx]['close'])))

        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        step += 1

    print(f"\nEpisode completed: {step} steps")
    final_asset = total_assets[-1] if total_assets else env.initial_balance
    profit = final_asset - env.initial_balance
    profit_pct = (profit / env.initial_balance) * 100

    print(f"Final asset: {final_asset:.2f} HKD")
    print(f"Total profit: {profit:.2f} HKD ({profit_pct:.2f}%)")

    # Count trades
    buys = sum(1 for a in actions if a == 1)
    sells = sum(1 for a in actions if a == 2)
    print(f"Trades: {buys} buys, {sells} sells")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(hspace=0.05)

    # Price chart with markers
    ax1.plot(dates, prices, 'k-', linewidth=1.5, alpha=0.8, label='Close Price')
    ax1.set_ylabel('Price (HKD)', fontsize=12)
    ax1.set_title('V1 Model Performance on Sideways Period (200 days, No Costs)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Mark trades
    buy_indices = [i for i, a in enumerate(actions) if a == 1]
    sell_indices = [i for i, a in enumerate(actions) if a == 2]

    if buy_indices:
        buy_dates = [dates[i] for i in buy_indices]
        buy_prices = [prices[i] for i in buy_indices]
        ax1.scatter(buy_dates, buy_prices, color='red', s=100, marker='^',
                   label='BUY', zorder=5, edgecolors='black', linewidth=1)

    if sell_indices:
        sell_dates = [dates[i] for i in sell_indices]
        sell_prices = [prices[i] for i in sell_indices]
        ax1.scatter(sell_dates, sell_prices, color='green', s=100, marker='v',
                   label='SELL', zorder=5, edgecolors='black', linewidth=1)

    ax1.legend(loc='upper left')

    # Asset curve
    ax2.plot(dates, total_assets, 'b-', linewidth=1.5, label='Total Asset')
    ax2.fill_between(dates, total_assets, alpha=0.3, color='blue')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Asset (HKD)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=env.initial_balance, color='red', linestyle='--',
                alpha=0.5, label='Initial')
    ax2.legend(loc='upper left')
    ax2.set_title(f'Final: {final_asset:.2f} HKD ({profit:+.2f}, {profit_pct:+.2f}%)',
                  fontsize=12)

    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = 'data/v1_sideways_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    # Action distribution
    print("\n" + "="*60)
    print("ACTION DISTRIBUTION")
    print("="*60)
    action_counts = pd.Series(actions).value_counts().sort_index()
    for act in [0, 1, 2]:
        count = action_counts.get(act, 0)
        pct = (count / len(actions)) * 100
        act_name = {0: 'Hold', 1: 'Buy', 2: 'Sell'}[act]
        print(f"  {act_name}: {count} ({pct:.1f}%)")

    plt.close()

if __name__ == "__main__":
    main()