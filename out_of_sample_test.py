"""
样本外测试（Out-of-Sample Test）
使用训练好的模型在"考卷"（2025-09至今）上回测
"""
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from env_v2 import RealisticHKEnv

os.environ['PYTHONPATH'] = '/app'

def main():
    print("="*70)
    print("OUT-OF-SAMPLE TEST (2025-09 ~ 2026-03)")
    print("="*70)

    # 加载在训练集上训练好的模型
    model_path = 'ppo_hkstock_v2_final.zip'
    try:
        model = PPO.load(model_path)
        print(f"✓ Loaded model from {model_path}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("Please run train_v2.py first to train the model.")
        return

    # 创建环境 - 使用测试集数据
    env = RealisticHKEnv(
        data_path='data/00700_test.csv',  # ⭐️ 测试集
        benchmark_path=None,
        use_features=True,
        enable_sharpe_penalty=False,  # 测试时关闭
        feature_window=30
    )

    # 运行回测
    obs, info = env.reset()
    done = False
    step = 0

    actions = []
    positions = []
    total_assets = []
    dates = []
    prices = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        action_int = int(action)

        actions.append(action_int)
        positions.append(env.position)
        total_assets.append(info['total_asset'])

        # 记录当前价格和日期
        current_idx = env.current_step - 1
        prices.append(env.df.iloc[current_idx]['close'])
        dates.append(env.df.iloc[current_idx]['date'])

        obs, reward, done, truncated, info = env.step(action)
        step += 1

    # 统计
    print(f"\n回测完成: {step} steps")
    initial_asset = total_assets[0]
    final_asset = total_assets[-1]
    profit = final_asset - initial_asset
    profit_pct = (profit / initial_asset) * 100

    print(f"初始资产: {initial_asset:.2f} HKD")
    print(f"最终资产: {final_asset:.2f} HKD")
    print(f"净利润: {profit:+.2f} HKD ({profit_pct:+.2f}%)")

    # 统计交易
    trades = 0
    for i in range(1, len(actions)):
        if actions[i] == 1 and actions[i-1] != 1:
            trades += 1
        elif actions[i] == 2 and actions[i-1] != 2:
            trades += 1

    buys = sum(1 for a in actions if a == 1)
    sells = sum(1 for a in actions if a == 2)
    holds = sum(1 for a in actions if a == 0)

    print(f"\n交易统计:")
    print(f"  总交易次数: {trades}")
    print(f"  买入: {buys}, 卖出: {sells}, 持有: {holds}")

    # ⚠️ 回撤检查
    print("\n" + "="*70)
    print("⚠️  回撤分析 (对比训练集预期表现)")
    print("="*70)

    # 加载训练集表现数据（从之前的测试结果）
    # 假设训练集上的表现是模型在训练环境上的回测（67280 HKD profit -> 338.9%）
    # 实际应该重新在训练集上跑一次获取准确数据，这里先计算测试集上的回撤
    # 从total_assets序列计算最大回撤
    assets_array = np.array(total_assets)
    running_max = np.maximum.accumulate(assets_array)
    drawdown = (assets_array - running_max) / running_max
    max_dd = np.min(drawdown) * 100  # 最大回撤百分比

    print(f"测试集最大回撤: {max_dd:.2f}%")

    # 回撤警报
    if abs(max_dd) > 20:
        print("🚨 警报: 样本外回撤超过 20%！")
        print("   建议: 检查模型泛化能力、市场环境变化、过拟合等")
    else:
        print("✅ 回撤控制在 20% 以内（但需对比训练集收益确认）")

    # 生成可视化
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(hspace=0.05)

    # 价格 + 交易标记
    ax1.plot(dates, prices, 'k-', linewidth=1.5, alpha=0.8, label='Close Price')
    ax1.set_ylabel('Price (HKD)', fontsize=12)
    ax1.set_title(f'OOD Test: 2025-09 to 2026-03 (Trades={trades}, Profit={profit_pct:+.1f}%)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Buy markers
    buy_indices = [i for i, a in enumerate(actions) if a == 1]
    if buy_indices:
        buy_dates = [dates[i] for i in buy_indices]
        buy_prices = [prices[i] for i in buy_indices]
        ax1.scatter(buy_dates, buy_prices, color='red', s=100, marker='^',
                   label='BUY', zorder=5, edgecolors='black', linewidth=1)

    # Sell markers
    sell_indices = [i for i, a in enumerate(actions) if a == 2]
    if sell_indices:
        sell_dates = [dates[i] for i in sell_indices]
        sell_prices = [prices[i] for i in sell_indices]
        ax1.scatter(sell_dates, sell_prices, color='green', s=100, marker='v',
                   label='SELL', zorder=5, edgecolors='black', linewidth=1)

    ax1.legend(loc='upper left')

    # 资产曲线
    ax2.plot(dates, total_assets, 'b-', linewidth=1.5, label='Total Asset')
    ax2.fill_between(dates, total_assets, alpha=0.3, color='blue')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Asset (HKD)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=initial_asset, color='red', linestyle='--', alpha=0.5, label='Initial')
    ax2.legend(loc='upper left')

    # 标注最大回撤区域
    if max_dd < -5:  # 如果回撤大于5%，标记出来
        dd_idx = np.argmin(drawdown)
        dd_date = dates[dd_idx]
        dd_value = assets_array[dd_idx]
        ax2.scatter(dd_date, dd_value, color='orange', s=200, marker='o',
                   label=f'Max DD: {max_dd:.1f}%', zorder=10, alpha=0.5)

    ax2.legend(loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = 'data/out_of_sample_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化已保存: {output_path}")

    # 对比训练集表现（需要从之前训练结果获取或重新跑训练集）
    print("\n" + "="*70)
    print("📊 训练集 vs 测试集 对比")
    print("="*70)
    print("训练集表现（v2模型，完整数据）: +338.90% (from earlier test)")
    print(f"测试集表现（2025-09至今）: {profit_pct:+.1f}%")
    performance_ratio = profit_pct / 338.90 if 338.90 != 0 else 0
    print(f"表现比例: {performance_ratio:.1%}")
    if performance_ratio < 0.8:  # 测试集收益不足训练集80%
        drop = (1 - performance_ratio) * 100
        print(f"⚠️  性能下降: {drop:.1f} 个百分点")
        if drop > 20:
            print("🚨 严重过拟合警报: 样本外收益比训练集低 20%+")
        else:
            print("ℹ️  性能有所下降，但在可接受范围")
    else:
        print("✅ 样本外表现良好，泛化能力较强")

    plt.close()

if __name__ == "__main__":
    main()
