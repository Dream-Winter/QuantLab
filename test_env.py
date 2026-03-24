import numpy as np
import pandas as pd
from env import HKStockEnv

def test_env():
    print("Testing HKStockEnv...")

    # Create environment
    env = HKStockEnv()

    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")

    # Run a few steps
    total_reward = 0
    for i in range(10):
        # Random action: 0 (hold), 1 (buy), 2 (sell)
        action = np.random.randint(0, 3)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        print(f"Step {i}: Action={action}, Reward={reward:.4f}, "
              f"Total Asset={info['total_asset']:.2f}, "
              f"Position={info['position']}, Balance={info['balance']:.2f}")

        if done:
            print("Episode finished!")
            break

    print(f"Total reward over {i+1} steps: {total_reward:.4f}")
    print("Environment test completed successfully!")

if __name__ == "__main__":
    test_env()