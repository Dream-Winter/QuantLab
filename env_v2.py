import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from feature_factory import FeatureFactory, generate_features


class RealisticHKEnv(gym.Env):
    """
    Industrial-grade Hong Kong stock trading environment with:
    - Realistic transaction costs (stamp duty, commission, fee)
    - Slippage simulation
    - Alpha-based reward (outperformance vs benchmark)
    - Sharpe ratio penalty for volatility
    - Feature factory for comprehensive technical factors
    """

    def __init__(
        self,
        data_path='data/00700_hist.csv',
        benchmark_path=None,  # Path to benchmark data (e.g., HSTECH)
        initial_balance=20000,
        transaction_fee_rate=0.002,  # 0.2% total fee (can be overridden below)
        stamp_duty=0.001,  # 0.1% stamp duty (sell only in HK)
        commission=0.0007,  # 0.07% commission
        slippage_buy=0.0005,  # +0.05% slippage on buy
        slippage_sell=-0.0005,  # -0.05% slippage on sell
        use_features=True,
        feature_window=30,  # Use features from past N days
        max_position_ratio=1.0,  # Max position as fraction of capital
        enable_sharpe_penalty=True,
        sharpe_window=20,
        risk_free_rate=0.02/252,  # Daily risk-free rate (~2% annual)
    ):
        super(RealisticHKEnv, self).__init__()

        # Load stock data
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)

        # Load benchmark data (恒生科技指数 for alpha calculation)
        self.use_benchmark = False
        self.benchmark_df = None
        if benchmark_path:
            try:
                self.benchmark_df = pd.read_csv(benchmark_path)
                self.benchmark_df['date'] = pd.to_datetime(self.benchmark_df['date'])
                self.benchmark_df = self.benchmark_df.sort_values('date').reset_index(drop=True)
                self.use_benchmark = True
                print(f"Loaded benchmark with {len(self.benchmark_df)} rows")
            except Exception as e:
                print(f"Warning: Could not load benchmark: {e}")
                print("Running without benchmark (absolute returns mode)")

        # Generate features if enabled
        self.use_features = use_features
        if use_features:
            print("Generating technical features...")
            factory = FeatureFactory()
            self.features_df = factory.calculate_all_features(self.df)
            self.feature_names = factory.get_feature_names()
            print(f"Generated {len(self.feature_names)} features")
        else:
            self.features_df = None
            self.feature_names = []

        # Observation window (how many past days to include)
        self.observation_window = feature_window  # Renamed for clarity

        # Cost parameters
        self.stamp_duty = stamp_duty  # Sell only
        self.commission = commission  # Both sides
        self.slippage_buy = slippage_buy
        self.slippage_sell = slippage_sell
        self.max_position_ratio = max_position_ratio

        # Sharpe penalty parameters
        self.enable_sharpe_penalty = enable_sharpe_penalty
        self.sharpe_window = sharpe_window
        self.risk_free_rate = risk_free_rate
        self.episode_returns = []  # Track returns for Sharpe calculation

        # Trading state
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares = 0.0
        self.position = 0  # 0: no position, 1: long
        self.entry_price = 0.0  # Price at which current position was opened
        self.current_step = 0
        self.max_steps = len(self.df) - 1

        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        # Observation space construction
        # Base: price/volume/RSI for past N days (3 * observation_window)
        base_dim = 3 * self.observation_window
        feature_dim = len(self.feature_names) * self.observation_window if use_features else 0
        total_dim = base_dim + feature_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )

        # Track total asset history for Sharpe calculation
        self.asset_history = []
        self.current_episode_rewards = []

    def _get_base_observation(self):
        """Get past N days of basic OHLCV + RSI data"""
        window_size = self.observation_window

        start_idx = max(0, self.current_step - window_size)
        end_idx = self.current_step

        window = self.df.iloc[start_idx:end_idx]

        if len(window) < window_size:
            # Pad with neutral values
            pad_len = window_size - len(window)
            close_pad = [window['close'].iloc[0] if len(window) > 0 else 0] * pad_len
            volume_pad = [window['volume'].iloc[0] if len(window) > 0 else 0] * pad_len
            rsi_pad = [50.0] * pad_len  # Neutral RSI

            closes = list(window['close']) + close_pad
            volumes = list(window['volume']) + volume_pad
            rsis = list(window.get('rsi', [50]*len(window))) + rsi_pad
        else:
            closes = list(window['close'])
            volumes = list(window['volume'])
            # Calculate RSI on demand if not present
            if 'rsi' in window.columns:
                rsis = list(window['rsi'])
            else:
                # Use ta library to compute RSI
                import ta
                rsi_series = ta.momentum.RSIIndicator(window['close'], window=14).rsi()
                rsis = list(rsi_series.fillna(50).values)

        # Normalize each series
        close_arr = np.array(closes)
        volume_arr = np.array(volumes)
        rsi_arr = np.array(rsis)

        close_norm = (close_arr - close_arr.min()) / (close_arr.max() - close_arr.min() + 1e-8)
        volume_norm = (volume_arr - volume_arr.min()) / (volume_arr.max() - volume_arr.min() + 1e-8)
        rsi_norm = rsi_arr / 100.0

        obs = []
        for i in range(self.observation_window):
            obs.extend([close_norm[i], volume_norm[i], rsi_norm[i]])

        return np.array(obs, dtype=np.float32)

    def _get_feature_observation(self):
        """Get concatenated feature vectors from past N days"""
        if not self.use_features:
            return np.array([], dtype=np.float32)

        start_idx = max(0, self.current_step - self.observation_window)
        end_idx = self.current_step

        feature_window = self.features_df.iloc[start_idx:end_idx]

        if len(feature_window) < self.observation_window:
            # Pad with zeros if not enough history
            pad_array = np.zeros((self.observation_window - len(feature_window), len(self.feature_names)))
            feature_window = pd.concat([
                pd.DataFrame(pad_array, columns=self.feature_names),
                feature_window
            ], ignore_index=True)

        # Flatten the feature matrix (time x features) to 1D
        obs = feature_window.values.flatten()
        return obs.astype(np.float32)

    def _get_total_asset(self, price):
        """Calculate total asset value (cash + position value)"""
        return self.balance + self.shares * price

    def _calculate_transaction_cost(self, price, shares, action):
        """
        Calculate total transaction cost including:
        - Commission (both buy and sell)
        - Stamp duty (sell only)
        - Slippage (price impact)
        """
        if action == 1:  # Buy
            # Slippage: pay slightly higher price
            exec_price = price * (1 + self.slippage_buy)
            # Commission: buyer pays
            gross_cost = shares * exec_price
            commission_cost = gross_cost * self.commission
            total_cost = gross_cost + commission_cost
            return exec_price, shares, total_cost  # Total cash needed

        elif action == 2:  # Sell
            # Slippage: receive slightly lower price
            exec_price = price * (1 + self.slippage_sell)
            # Commission + Stamp duty: seller pays both
            gross_proceeds = shares * exec_price
            commission_cost = gross_proceeds * self.commission
            stamp_duty_cost = gross_proceeds * self.stamp_duty
            net_cash_in = gross_proceeds - commission_cost - stamp_duty_cost
            return exec_price, shares, net_cash_in  # Positive amount received

        return price, 0, 0

    def _get_benchmark_return(self, current_idx, lookahead=1):
        """Get benchmark (恒生科技指数) return for alpha calculation"""
        if not self.use_benchmark or self.benchmark_df is None:
            return 0.0

        if current_idx + lookahead >= len(self.benchmark_df):
            return 0.0

        curr_price = self.benchmark_df.iloc[current_idx]['close']
        next_price = self.benchmark_df.iloc[current_idx + lookahead]['close']
        return (next_price - curr_price) / curr_price if curr_price > 0 else 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset trading state
        self.balance = self.initial_balance
        self.shares = 0.0
        self.position = 0
        self.entry_price = 0.0
        self.entry_total_asset = None  # Asset value at position open
        self.current_step = 5  # Start after enough data for observation

        # Reset tracking
        self.asset_history = [self.initial_balance]
        self.current_episode_rewards = []

        observation = self._get_observation()
        info = {
            'balance': self.balance,
            'shares': self.shares,
            'position': self.position,
            'total_asset': self.initial_balance
        }

        return observation, info

    def _get_observation(self):
        """Concatenate base and feature observations"""
        base_obs = self._get_base_observation()
        if self.use_features:
            feature_obs = self._get_feature_observation()
            return np.concatenate([base_obs, feature_obs])
        return base_obs

    def step(self, action):
        # Current price before action
        current_price_raw = self.df.iloc[self.current_step]['close']
        prev_total_asset = self._get_total_asset(current_price_raw)
        prev_position = self.position  # ⭐️ 记录执行前的持仓状态

        # Track if trade executed
        executed_trade = False

        # Execute action
        if action == 1:  # Buy
            if self.position == 0:  # Only if no position
                executed_trade = True
                # Calculate the actual number of shares we can afford including commission
                exec_price_approx = current_price_raw * (1 + self.slippage_buy)
                cost_per_share = exec_price_approx * (1 + self.commission)
                affordable_shares = self.balance / cost_per_share
                shares_to_buy = affordable_shares * self.max_position_ratio

                exec_price, shares_bought, cash_out = self._calculate_transaction_cost(
                    current_price_raw, shares_to_buy, action
                )

                if cash_out <= self.balance and shares_bought > 0:
                    self.shares = shares_bought
                    self.balance = self.balance - cash_out
                    self.position = 1
                    self.entry_price = exec_price
                    self.entry_total_asset = prev_total_asset  # ⭐️ 记录开仓时的资产
                else:
                    executed_trade = False

        elif action == 2:  # Sell
            if self.position == 1:  # Only if have position
                executed_trade = True
                shares_to_sell = self.shares
                exec_price, shares_sold, cash_in = self._calculate_transaction_cost(
                    current_price_raw, shares_to_sell, action
                )
                self.balance = self.balance + cash_in
                self.shares = 0.0
                self.position = 0
                self.entry_price = 0.0

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        if not done:
            next_price = self.df.iloc[self.current_step]['close']
        else:
            next_price = self.df.iloc[self.max_steps]['close']

        current_total_asset = self._get_total_asset(next_price)
        self.asset_history.append(current_total_asset)

        # --- ⭐️ REWARD CALCULATION (Refactored: Settlement-based) ---
        reward = 0.0

        if action == 0:  # Hold/Wait
            if prev_position == 1:  # 持仓期间：微小时间成本，防止死拿
                reward = -0.0001
            else:  # 空仓观望：奖励为0
                reward = 0.0

        elif action == 1:  # Buy
            if executed_trade:
                reward = 0.001  # 开仓微小正奖励（鼓励建仓）
            else:
                reward = -0.01  # 无效买入惩罚

        elif action == 2:  # Sell
            if executed_trade:
                # 平仓结算奖励：(当前资产 - 买入时资产) / 初始资金
                if hasattr(self, 'entry_total_asset') and self.entry_total_asset is not None:
                    profit = current_total_asset - self.entry_total_asset
                    reward = profit / self.initial_balance
                    # 归一化：确保绝对值在 0.1 以内（理论上已满足，加个保险）
                    reward = np.clip(reward, -0.1, 0.1)
                else:
                    reward = 0.0
                self.entry_total_asset = None  # 清空开仓记录
            else:
                reward = -0.01  # 无效卖出惩罚

        # 强制归一化（保险）
        reward = np.clip(reward, -0.1, 0.1)

        # Get next observation
        if not done:
            observation = self._get_observation()
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            'total_asset': current_total_asset,
            'balance': self.balance,
            'shares': self.shares,
            'position': self.position,
            'price': next_price,
            'action': action,
            'reward': reward,
            'executed_trade': executed_trade,
            'episode_progress': self.current_step / self.max_steps
        }

        return observation, reward, done, False, info

    def render(self):
        pass

    def close(self):
        pass