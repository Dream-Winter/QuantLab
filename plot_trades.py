import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Docker
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from env import HKStockEnv

def main():
    print("Loading trained model and environment...")

    # Load the trained model
    model = PPO.load('ppo_hkstock_final')

    # Create environment
    env = HKStockEnv()
    obs, _ = env.reset()

    # Run a complete test episode
    prices = []
    actions = []
    total_assets = []
    dates = []

    done = False
    step = 0

    while not done:
        # Get deterministic action
        action, _states = model.predict(obs, deterministic=True)

        # Record current price and date
        current_data = env.df.iloc[env.current_step - 1]  # Current day's data
        prices.append(current_data['close'])
        dates.append(current_data['date'])
        actions.append(action)

        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        total_assets.append(info['total_asset'])
        step += 1

    print(f"Episode completed in {step} steps")
    print(f"Total trading actions (buy/sell): {sum(1 for a in actions if a != 0)}")

    # Count trades
    trades = 0
    for i in range(1, len(actions)):
        if actions[i] == 1 and actions[i-1] != 1:  # Buy action (entering position)
            trades += 1
        elif actions[i] == 2 and actions[i-1] != 2:  # Sell action (exiting position)
            trades += 1

    print(f"Number of trades (buy/sell pairs): {trades}")
    print(f"Final total asset: {info['total_asset']:.2f} HKD")

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(hspace=0.05)

    # Plot 1: Price with trade markers
    ax1.plot(dates, prices, 'k-', linewidth=1.5, alpha=0.8, label='Close Price')
    ax1.set_ylabel('Price (HKD)', fontsize=12)
    ax1.set_title('Trading Behavior Visualization - Tencent (00700.HK)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Mark buy points (Action 1)
    buy_indices = [i for i, a in enumerate(actions) if a == 1]
    if buy_indices:
        buy_dates = [dates[i] for i in buy_indices]
        buy_prices = [prices[i] for i in buy_indices]
        ax1.scatter(buy_dates, buy_prices, color='red', s=100, marker='^',
                   label='BUY (Action 1)', zorder=5, edgecolors='black', linewidth=1)

    # Mark sell points (Action 2)
    sell_indices = [i for i, a in enumerate(actions) if a == 2]
    if sell_indices:
        sell_dates = [dates[i] for i in sell_indices]
        sell_prices = [prices[i] for i in sell_indices]
        ax1.scatter(sell_dates, sell_prices, color='green', s=100, marker='v',
                   label='SELL (Action 2)', zorder=5, edgecolors='black', linewidth=1)

    ax1.legend(loc='upper left')

    # Plot 2: Total asset curve
    ax2.plot(dates, total_assets, 'b-', linewidth=1.5, label='Total Asset')
    ax2.fill_between(dates, total_assets, alpha=0.3, color='blue')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Asset (HKD)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    # Highlight profit regions
    if total_assets:
        initial_asset = total_assets[0]
        final_asset = total_assets[-1]
        profit = final_asset - initial_asset
        profit_pct = (profit / initial_asset) * 100

        ax2.axhline(y=initial_asset, color='gray', linestyle='--', alpha=0.5, label='Initial')
        ax2.set_title(f'Final: {final_asset:.2f} HKD ({profit:+.2f}, {profit_pct:+.2f}%)',
                     fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Save figure
    output_path = 'data/trading_behavior.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nTrading visualization saved to: {output_path}")

    # Show summary statistics
    print("\n" + "="*50)
    print("TRADING SUMMARY")
    print("="*50)
    print(f"Total Steps: {step}")
    print(f"Trades Executed: {trades}")
    print(f"Buy Signals: {len(buy_indices)}")
    print(f"Sell Signals: {len(sell_indices)}")
    print(f"Initial Asset: {total_assets[0]:.2f} HKD")
    print(f"Final Asset: {final_asset:.2f} HKD")
    print(f"Net Profit: {profit:+.2f} HKD ({profit_pct:+.2f}%)")
    print("="*50)

    plt.close()

if __name__ == "__main__":
    main()