import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class HKStockEnv(gym.Env):
    """
    Custom Trading Environment for Hong Kong Stocks
    """
    def __init__(self, data_path='data/00700_hist.csv', initial_balance=20000,
                 transaction_fee_rate=0.002, recent_days=None):
        super(HKStockEnv, self).__init__()

        # Load data
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)

        # Optionally use only recent data for training
        if recent_days is not None and recent_days < len(self.df):
            self.df = self.df.tail(recent_days).reset_index(drop=True)
            print(f"Using recent {recent_days} days of data ({len(self.df)} rows)")
        else:
            print(f"Using full dataset ({len(self.df)} rows)")

        # Technical indicators
        self.df['rsi'] = self._calculate_rsi(self.df['close'])

        # Normalization parameters (using entire dataset for simplicity)
        self.close_max = self.df['close'].max()
        self.close_min = self.df['close'].min()
        self.volume_max = self.df['volume'].max()
        self.volume_min = self.df['volume'].min()

        # Action space: 0 (hold), 1 (buy), 2 (sell)
        self.action_space = spaces.Discrete(3)

        # Observation space: past 5 days of [close, volume, rsi] -> 5*3 = 15 features
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(15,), dtype=np.float32
        )

        # Trading parameters
        self.initial_balance = initial_balance
        self.transaction_fee_rate = transaction_fee_rate  # 0.2% transaction fee
        self.balance = initial_balance
        self.shares = 0
        self.position = 0  # 0: no position, 1: long
        self.entry_price = 0
        self.current_step = 0
        self.max_steps = len(self.df) - 1

        # For reward calculation
        self.prev_total_asset = self.initial_balance

    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral 50

    def _normalize_close(self, close):
        """Normalize close price to [0, 1]"""
        if self.close_max == self.close_min:
            return 0.5
        return (close - self.close_min) / (self.close_max - self.close_min)

    def _normalize_volume(self, volume):
        """Normalize volume to [0, 1]"""
        if self.volume_max == self.volume_min:
            return 0.5
        return (volume - self.volume_min) / (self.volume_max - self.volume_min)

    def _normalize_rsi(self, rsi):
        """Normalize RSI to [0, 1]"""
        return rsi / 100.0

    def _get_observation(self):
        """Get observation of past 5 days"""
        # Start from current_step - 5 to current_step - 1 (5 days ago to yesterday)
        start_idx = max(0, self.current_step - 5)
        end_idx = self.current_step

        # Extract the window
        window = self.df.iloc[start_idx:end_idx]

        # If we don't have enough data, pad with zeros
        if len(window) < 5:
            padding = pd.DataFrame([
                {'close': 0, 'volume': 0, 'rsi': 50}  # Neutral RSI
                for _ in range(5 - len(window))
            ])
            window = pd.concat([padding, window], ignore_index=True)

        # Extract features and normalize
        closes = [self._normalize_close(c) for c in window['close']]
        volumes = [self._normalize_volume(v) for v in window['volume']]
        rsis = [self._normalize_rsi(r) for r in window['rsi']]

        # Flatten to 1D array: [close1, volume1, rsi1, close2, volume2, rsi2, ...]
        obs = []
        for i in range(5):
            obs.extend([closes[i], volumes[i], rsis[i]])

        return np.array(obs, dtype=np.float32)

    def _get_total_asset(self, close_price):
        """Calculate total asset value (cash + shares value)"""
        return self.balance + self.shares * close_price

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        # Reset trading state
        self.balance = self.initial_balance
        self.shares = 0
        self.position = 0
        self.entry_price = 0
        self.current_step = 5  # Start after we have 5 days of data for observation
        self.prev_total_asset = self.initial_balance

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """Execute one time step within the environment"""
        # Get current price (today's close)
        current_close = self.df.iloc[self.current_step]['close']

        # Store previous state for reward calculation
        prev_balance = self.balance
        prev_shares = self.shares
        prev_position = self.position
        prev_total_asset = self._get_total_asset(current_close)

        # Execute action
        # Track if we executed a valid trade
        executed_trade = False

        if action == 1:  # Buy
            if self.position == 0:  # Only buy if no position
                executed_trade = True
                # Calculate transaction fee (0.2%)
                fee_rate = self.transaction_fee_rate
                # Buy shares with fee included: shares = balance / (price * (1 + fee_rate))
                cost_per_share_with_fee = current_close * (1 + fee_rate)
                self.shares = self.balance / cost_per_share_with_fee
                self.balance = 0
                self.position = 1
                self.entry_price = current_close
        elif action == 2:  # Sell
            if self.position == 1:  # Only sell if we have position
                executed_trade = True
                # Sell all shares with transaction fee (0.2%)
                fee_rate = self.transaction_fee_rate
                self.balance = self.shares * current_close * (1 - fee_rate)
                self.shares = 0
                self.position = 0
                self.entry_price = 0
        # action == 0: Hold (do nothing)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= self.max_steps

        # Get new total asset value (using today's close after action)
        if not done:
            current_close = self.df.iloc[self.current_step]['close']
        else:
            # If done, use the last available close
            current_close = self.df.iloc[self.max_steps]['close']

        current_total_asset = self._get_total_asset(current_close)

        # Calculate reward based on unrealized P&L when holding position
        # This decouples entry cost from ongoing performance
        if self.position == 1:
            # Unrealized return from entry price to current price
            if self.entry_price > 0:
                unrealized_return = (current_close - self.entry_price) / self.entry_price
                reward = unrealized_return * 100  # Scale
            else:
                reward = 0
        else:
            # No position: no market-based reward
            reward = 0

        # Small penalty for invalid actions
        if action == 1 and self.position != 0:  # Invalid buy (already long)
            reward -= 0.5
        elif action == 2 and self.position != 1:  # Invalid sell (no position)
            reward -= 0.5

        # Transaction trade penalty (applied only to valid trades)
        if executed_trade:
            reward -= 0.2  # Small penalty for turning over capital

        # Additional penalty for holding during significant drawdowns
        if (self.position == 1 and
            prev_position == 1 and
            current_close < self.df.iloc[self.current_step - 1]['close']):
            down_pct = (self.df.iloc[self.current_step - 1]['close'] - current_close) / self.df.iloc[self.current_step - 1]['close']
            reward -= down_pct * 10  # Moderate drawdown penalty

        # Get next observation
        if not done:
            observation = self._get_observation()
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        # Info for debugging
        info = {
            'total_asset': current_total_asset,
            'balance': self.balance,
            'shares': self.shares,
            'position': self.position,
            'current_close': current_close,
            'action': action,
            'reward': reward
        }

        # Update previous total asset for next step
        self.prev_total_asset = current_total_asset

        return observation, reward, done, False, info

    def render(self):
        """Render the environment (optional)"""
        pass

    def close(self):
        """Close the environment (optional)"""
        pass