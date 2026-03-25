import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from env_v2 import RealisticHKEnv

os.environ['PYTHONPATH'] = '/app'

def make_env():
    """Create simplified trading environment"""
    env = RealisticHKEnv(
        data_path='data/00700_hist.csv',
        benchmark_path=None,  # No benchmark
        initial_balance=20000,
        use_features=False,  # No complex features
        enable_sharpe_penalty=False,  # No Sharpe penalty
        feature_window=5,
        stamp_duty=0.001,
        commission=0.0007,
        slippage_buy=0.0005,
        slippage_sell=-0.0005,
        max_position_ratio=1.0
    )
    return env

def main():
    print("Starting SIMPLIFIED training (no features, no Sharpe)...")
    print("=" * 60)

    # Create environments
    env = make_vec_env(make_env, n_envs=1)
    eval_env = make_env()

    # Callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/best_model_simple',
        log_path='./logs_simple/',
        eval_freq=1000,
        deterministic=True,
        render=False,
        n_eval_episodes=1
    )

    # PPO agent
    print("Initializing PPO...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log='./tensorboard_simple/'
    )

    # Train
    print("Training (20,000 steps)...")
    total_timesteps = 20000
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name='ppo_simple'
    )

    model.save('ppo_hkstock_simple_final')
    print("Model saved to ppo_hkstock_simple_final")

    # Evaluate
    print("\nEvaluating...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=3)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Test episode
    print("\nTest episode:")
    obs = eval_env.reset()[0]
    total_reward = 0
    done = False
    step = 0
    actions = []
    while not done and step < 200:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, truncated, info = eval_env.step(action)
        total_reward += reward
        step += 1
    print(f"Steps: {step}, Final asset: {info['total_asset']:.2f}, Total reward: {total_reward:.2f}")
    print(f"Actions: {actions[:50]}...")

if __name__ == "__main__":
    main()