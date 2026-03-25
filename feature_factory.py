"""
Feature Factory for Hong Kong Stock Trading
Implements industrial-grade factor generation (Alpha158-like subset)
"""

import numpy as np
import pandas as pd
import ta  # Technical Analysis library (pure Python TA-Lib alternative)


class FeatureFactory:
    """
    Generate technical indicators and alpha factors for HK stocks.
    All features are Z-score normalized to avoid scale issues.
    """

    def __init__(self, window_sizes=[5, 10, 20, 60, 120]):
        self.window_sizes = window_sizes
        self.feature_names = []

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive set of technical factors.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame with normalized features
        """
        df = df.copy()
        features = pd.DataFrame(index=df.index)

        # 1. Price-based momentum indicators
        self._add_momentum_factors(df, features)

        # 2. Volatility factors
        self._add_volatility_factors(df, features)

        # 3. Volume-based factors
        self._add_volume_factors(df, features)

        # 4. Trend factors
        self._add_trend_factors(df, features)

        # 5. Oscillator factors
        self._add_oscillator_factors(df, features)

        # 6. Pattern recognition (candlestick patterns)
        self._add_pattern_factors(df, features)

        # 7. Stat arb factors (cross-sectional ranks)
        self._add_stat_arb_factors(df, features)

        # Z-score normalization (using rolling mean/std to avoid lookahead)
        features = self._normalize_features(features, df)

        return features

    def _add_momentum_factors(self, df: pd.DataFrame, features: pd.DataFrame):
        """Momentum class: RSI, MACD, CCI, ROC, STOCH, etc."""
        close = df['close']
        high = df['high']
        low = df['low']

        # RSI (Relative Strength Index)
        for period in [6, 12, 24]:
            rsi = ta.momentum.RSIIndicator(close, window=period).rsi()
            features[f'rsi_{period}'] = rsi

        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(close)
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        features['macd_diff'] = macd.macd_diff()

        # CCI (Commodity Channel Index)
        for period in [14, 20]:
            cci = ta.trend.CCIIndicator(high, low, close, window=period).cci()
            features[f'cci_{period}'] = cci

        # ROC (Rate of Change)
        for period in [10, 20]:
            roc = ta.momentum.ROCIndicator(close, window=period).roc()
            features[f'roc_{period}'] = roc

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14)
        features['stoch_k'] = stoch.stoch()
        features['stoch_d'] = stoch.stoch_signal()

        # Williams %R
        for period in [10, 14]:
            wr = ta.momentum.WilliamsRIndicator(high, low, close, lbp=period).williams_r()
            features[f'williams_r_{period}'] = wr

    def _add_volatility_factors(self, df: pd.DataFrame, features: pd.DataFrame):
        """Volatility class: ATR, Bollinger Bands, Standard Deviation"""
        high = df['high']
        low = df['low']
        close = df['close']

        # ATR (Average True Range)
        for period in [14, 20]:
            atr = ta.volatility.AverageTrueRange(high, low, close, window=period).average_true_range()
            features[f'atr_{period}'] = atr

        # Bollinger Bands
        for period in [20, 30]:
            bb = ta.volatility.BollingerBands(close, window=period)
            features[f'bb_upper_{period}'] = bb.bollinger_hband()
            features[f'bb_lower_{period}'] = bb.bollinger_lband()
            bb_middle = bb.bollinger_mavg()
            features[f'bb_width_{period}'] = (bb.bollinger_hband() - bb.bollinger_lband()) / (bb_middle + 1e-8)
            features[f'bb_percent_{period}'] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)

        # Standard deviation of returns
        returns = close.pct_change()
        for period in [10, 20, 60]:
            vol = returns.rolling(period).std() * np.sqrt(252)  # Annualized
            features[f'volatility_{period}'] = vol

    def _add_volume_factors(self, df: pd.DataFrame, features: pd.DataFrame):
        """Volume class: OBV, Volume SMA, Volume Price Confirmation"""
        close = df['close']
        volume = df['volume']

        # OBV (On Balance Volume)
        obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        features['obv'] = obv

        # Volume SMA and ratio
        for period in [5, 10, 20]:
            vol_sma = volume.rolling(period).mean()
            features[f'volume_sma_{period}'] = vol_sma
            features[f'volume_ratio_{period}'] = volume / (vol_sma + 1e-8)

        # Volume Price Trend (VPT)
        vpt = ta.volume.VolumePriceTrendIndicator(close, volume).volume_price_trend()
        features['vpt'] = vpt

        # Money Flow Index (MFI)
        high = df['high']
        low = df['low']
        for period in [14, 21]:
            mfi = ta.volume.MFIIndicator(high, low, close, volume, window=period).money_flow_index()
            features[f'mfi_{period}'] = mfi

    def _add_trend_factors(self, df: pd.DataFrame, features: pd.DataFrame):
        """Trend class: SMA, EMA, ADX, Parabolic SAR"""
        close = df['close']
        high = df['high']
        low = df['low']

        # Simple Moving Averages
        for period in [5, 10, 20, 60, 120]:
            sma = ta.trend.SMAIndicator(close, window=period).sma_indicator()
            ema = ta.trend.EMAIndicator(close, window=period).ema_indicator()
            features[f'sma_{period}'] = sma
            features[f'ema_{period}'] = ema
            # Price to MA ratio
            features[f'price_to_sma_{period}'] = close / (sma + 1e-8)
            features[f'price_to_ema_{period}'] = close / (ema + 1e-8)
            # MA cross signals (short vs long)
            if period in [20, 60]:
                features[f'sma_20_60_ratio'] = sma / (features.get('sma_60', sma) + 1e-8)

        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(high, low, close, window=14)
        features['adx'] = adx.adx()
        features['adx_pos'] = adx.adx_pos()
        features['adx_neg'] = adx.adx_neg()

        # Parabolic SAR
        psar = ta.trend.PSARIndicator(high, low, close)
        features['psar'] = psar.psar()
        features['psar_up'] = psar.psar_up_indicator()
        features['psar_down'] = psar.psar_down_indicator()

        # Ichimoku Cloud (simplified)
        ichimoku = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
        features['ichimoku_a'] = ichimoku.ichimoku_a()
        features['ichimoku_b'] = ichimoku.ichimoku_b()
        features['ichimoku_base'] = ichimoku.ichimoku_base_line()
        features['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()

    def _add_oscillator_factors(self, df: pd.DataFrame, features: pd.DataFrame):
        """Oscillator factors"""
        close = df['close']
        high = df['high']
        low = df['low']

        # Awesome Oscillator
        ao = ta.momentum.AwesomeOscillatorIndicator(high, low).awesome_oscillator()
        features['awesome_oscillator'] = ao

        # KAMA (Kaufman's Adaptive Moving Average)
        kama = ta.momentum.KAMAIndicator(close).kama()
        features['kama'] = kama
        features['price_to_kama'] = close / (kama + 1e-8)

        # PPO (Percentage Price Oscillator)
        ppo = ta.momentum.PercentagePriceOscillator(close)
        features['ppo'] = ppo.ppo()
        features['ppo_signal'] = ppo.ppo_signal()
        features['ppo_diff'] = ppo.ppo_hist()

        # TRIX (Triple Exponential Moving Average)
        trix = ta.trend.TRIXIndicator(close).trix()
        features['trix'] = trix

    def _add_pattern_factors(self, df: pd.DataFrame, features: pd.DataFrame):
        """Candlestick pattern recognition (simplified)"""
        # Due to ta library limitations, we'll skip complex pattern recognition
        # In production, use TA-Lib or specialized pattern libraries
        # For now, leave this section empty or add simple heuristics
        pass

    def _add_stat_arb_factors(self, df: pd.DataFrame, features: pd.DataFrame):
        """Statistical arbitrage factors"""
        close = df['close']
        volume = df['volume']

        # Price ranks over rolling windows
        for window in [10, 20, 60]:
            rolling_min = close.rolling(window).min()
            rolling_max = close.rolling(window).max()
            price_rank = (close - rolling_min) / (rolling_max - rolling_min + 1e-8)
            features[f'price_rank_{window}'] = price_rank

        # Volume-Price correlation
        for window in [10, 20]:
            corr = close.rolling(window).corr(volume)
            features[f'price_volume_corr_{window}'] = corr

        # Z-score of price (mean reversion signal)
        for window in [20, 60]:
            rolling_mean = close.rolling(window).mean()
            rolling_std = close.rolling(window).std()
            zscore = (close - rolling_mean) / (rolling_std + 1e-8)
            features[f'price_zscore_{window}'] = zscore.clip(-5, 5)  # Clip outliers

    def _normalize_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score normalize features using rolling statistics (no lookahead bias).
        For production, these stats should be precomputed and saved.
        """
        normalized_features = features.copy()

        for col in features.columns:
            series = features[col]
            # Use rolling mean/std with 60-day window
            rolling_mean = series.rolling(60, min_periods=20).mean()
            rolling_std = series.rolling(60, min_periods=20).std()

            # Z-score
            normalized_features[col] = (series - rolling_mean) / (rolling_std + 1e-8)

            # Clip extreme values to [-3, 3]
            normalized_features[col] = normalized_features[col].clip(-3, 3)

        # Fill remaining NaNs with 0 (neutral)
        normalized_features = normalized_features.fillna(0)

        self.feature_names = list(normalized_features.columns)
        return normalized_features

    def get_feature_names(self) -> list:
        """Return list of feature names"""
        return self.feature_names


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to generate features.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with normalized features (same index as input)
    """
    factory = FeatureFactory()
    return factory.calculate_all_features(df)


if __name__ == "__main__":
    # Test the feature factory
    import sys
    sys.path.insert(0, '/app')

    df = pd.read_csv('data/00700_hist.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"Input data: {len(df)} rows")

    factory = FeatureFactory()
    features = factory.calculate_all_features(df)

    print(f"\nGenerated {len(features.columns)} features:")
    print(features.columns.tolist())
    print(f"\nFeature stats:")
    print(features.describe().T[['mean', 'std', 'min', 'max']])

    # Save features
    features.to_csv('data/00700_features.csv', index=False)
    print(f"\nFeatures saved to data/00700_features.csv")