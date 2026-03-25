import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from env_v2 import RealisticHKEnv

# Set PYTHONPATH for module imports
os.environ['PYTHONPATH'] = '/app'

def make_env():
    """Create the trading environment"""
    # Try to use benchmark if available
    benchmark_path = 'data/synthetic_benchmark.csv' if os.path.exists('data/synthetic_benchmark.csv') else None
    env = RealisticHKEnv(
        data_path='data/00700_train.csv',  # ⭐️ 使用训练集
        benchmark_path=benchmark_path,
        initial_balance=20000,
        use_features=True,  # Enable feature factory
        enable_sharpe_penalty=True,
        sharpe_window=20
    )
    return env

def main():
    print("Starting RL training with RealisticHKEnv v2...")
    print("=" * 60)

    # Create vectorized environment
    env = make_vec_env(make_env, n_envs=1)

    # Create evaluation environment
    eval_env = make_env()

    # Callback for evaluation during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/best_model_v2',
        log_path='./logs_v2/',
        eval_freq=2000,
        deterministic=True,
        render=False,
        n_eval_episodes=1
    )

    # Create PPO agent
    print("Initializing PPO agent...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=1,
        tensorboard_log='./tensorboard_v2/'
    )

    # Train the agent
    print("Starting training with features and realistic costs...")
    total_timesteps = 100000
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name='ppo_hkstock_v2'
    )

    # Save the model
    model.save('ppo_hkstock_v2_final')
    print(f"Model saved to ppo_hkstock_v2_final")

    # Evaluate the trained agent
    print("\nEvaluating agent...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Run a test episode with visualization
    print("\nRunning test episode...")
    obs = eval_env.reset()[0]
    total_reward = 0
    done = False
    step = 0

    actions = []
    prices = []
    assets = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, truncated, info = eval_env.step(action)
        total_reward += reward
        prices.append(info['price'])
        assets.append(info['total_asset'])
        step += 1

    print(f"\nEpisode finished after {step} steps")
    print(f"Final total asset: {info['total_asset']:.2f} HKD")
    print(f"Initial balance: {eval_env.initial_balance:.2f} HKD")
    profit = info['total_asset'] - eval_env.initial_balance
    profit_pct = (profit / eval_env.initial_balance) * 100
    print(f"Total profit/loss: {profit:.2f} HKD ({profit_pct:.2f}%)")

    # Count trades
    trades = 0
    for i in range(1, len(actions)):
        if actions[i] == 1 and actions[i-1] != 1:  # Buy
            trades += 1
        elif actions[i] == 2 and actions[i-1] != 2:  # Sell
            trades += 1

    print(f"Number of trades: {trades}")

    # Save results for visualization
    results_df = pd.DataFrame({
        'step': range(len(prices)),
        'price': prices,
        'total_asset': assets,
        'action': actions[:len(prices)]
    })
    results_df.to_csv('data/v2_test_results.csv', index=False)
    print(f"\nTest results saved to data/v2_test_results.csv")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("Next steps: run plot_trades_v2.py to visualize")
    print("=" * 60)

if __name__ == "__main__":
    main()