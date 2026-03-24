import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from env import HKStockEnv

# Set PYTHONPATH for module imports
os.environ['PYTHONPATH'] = '/app'

def make_env():
    """Create the trading environment"""
    env = HKStockEnv()
    return env

def main():
    print("Starting RL training for HK stock trading...")

    # Create vectorized environment (1 env for now)
    env = make_vec_env(make_env, n_envs=1)

    # Create evaluation environment
    eval_env = HKStockEnv()

    # Callback for evaluation during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/best_model',
        log_path='./logs/',
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    # Create PPO agent
    print("Initializing PPO agent...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=4096,  # Larger batch for longer episodes
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log='./tensorboard/'
    )

    # Train the agent
    print("Starting training...")
    total_timesteps = 100000  # More steps for long episodes
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name='ppo_hkstock'
    )

    # Save the model
    model.save('ppo_hkstock_final')
    print(f"Model saved to ppo_hkstock_final")

    # Evaluate the trained agent
    print("Evaluating agent...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Run a quick test episode to visualize performance
    print("\nRunning test episode...")
    obs = eval_env.reset()[0]
    total_reward = 0
    done = False
    step = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        total_reward += reward
        step += 1
        if step % 10 == 0:
            print(f"Step {step}: Action={action}, Reward={reward:.4f}, "
                  f"Total Asset={info['total_asset']:.2f}")

    print(f"\nEpisode finished after {step} steps")
    print(f"Final total asset: {info['total_asset']:.2f} HKD")
    print(f"Initial balance: {eval_env.initial_balance:.2f} HKD")
    print(f"Total profit/loss: {info['total_asset'] - eval_env.initial_balance:.2f} HKD")

if __name__ == "__main__":
    main()