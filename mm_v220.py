"""
Multi_v210
V2 Script for Hummingbot - Raspberry Pi Optimized, Kraken fee = 0.26%!
"""
import sys
import os

scripts_path = '/home/hummingbot/scripts'
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)
    print(f"✅ Added {scripts_path} to Python path")

from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.event.events import OrderType as EventOrderType
from decimal import Decimal
import time
import statistics
from collections import deque
from enum import Enum
from typing import Dict, Tuple, Optional, Any
import logging
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NOW import Tickr - after path is fixed
TICKR_AVAILABLE = False
try:
    from tickr import TickrClient
    TICKR_AVAILABLE = True
    logger.info("✅ Tickr successfully imported (local module)")
    
    # Test instantiation
    try:
        test_client = TickrClient(exchange="kraken")
        logger.info("✅ Tickr client instantiation successful")
        del test_client
    except Exception as e:
        logger.error(f"⚠️ Tickr client instantiation failed: {e}")
        TICKR_AVAILABLE = False
        
except ImportError as e:
    logger.error(f"❌ Tickr import failed - Error: {e}")
    logger.error(f"Python path: {sys.path[:5]}")
    logger.error(f"Scripts directory exists: {os.path.exists('/home/hummingbot/scripts/tickr.py')}")
    TICKR_AVAILABLE = False
except Exception as e:
    logger.error(f"❌ Unexpected error importing Tickr: {e}", exc_info=True)
    TICKR_AVAILABLE = False

# TA-Lib for technical analysis (optional but recommended)
try:
    import talib
    import pandas as pd
    import numpy as np
    TALIB_AVAILABLE = True
    logger.info("✅ TA-Lib successfully imported")
except ImportError as e:
    TALIB_AVAILABLE = False
    logger.warning(f"⚠️ TA-Lib not available - Error: {e}")


# ==================== ENUMS ====================
class MarketRegime(Enum):
    # MODIFIED: Removed unused HALTED value
    ULTRA_CALM = "ultra_calm"
    CALM = "calm"
    NORMAL = "normal"
    VOLATILE = "volatile"
    EXTREME = "extreme"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CHOPPY = "choppy"

class InventoryZone(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    CRITICAL = "critical"

# ==================== CONFIGURATIONS ====================
# ==================== CONFIGURATIONS ====================
class BaseConfig:
    def __init__(self, **kwargs):
        # Core parameters
        self.absolute_min_spread = kwargs.get('absolute_min_spread', 0.0026)
        self.absolute_max_spread = kwargs.get('absolute_max_spread', 0.02)
        self.target_fee_coverage = kwargs.get('target_fee_coverage', 1.5)
        self.minimum_fee_coverage = kwargs.get('minimum_fee_coverage', 1.15)
        
        # Technical analysis
        self.volatility_window = kwargs.get('volatility_window', 20)
        self.fast_ma_window = kwargs.get('fast_ma_window', 8)
        self.slow_ma_window = kwargs.get('slow_ma_window', 24)
        self.momentum_window = kwargs.get('momentum_window', 6)
        
        # Market sensitivity
        self.volatility_extreme_threshold = kwargs.get('volatility_extreme_threshold', 0.002)
        self.volatility_high_threshold = kwargs.get('volatility_high_threshold', 0.001)
        self.volatility_low_threshold = kwargs.get('volatility_low_threshold', 0.0003)
        self.volatility_ultra_low_threshold = kwargs.get('volatility_ultra_low_threshold', 0.00015)
        
        # Trend detection
        self.trend_strong_threshold = kwargs.get('trend_strong_threshold', 0.00075)
        self.momentum_strong_threshold = kwargs.get('momentum_strong_threshold', 0.0005)
        
        # Inventory management
        self.inventory_target_pct = kwargs.get('inventory_target_pct', 50.0)
        # MODIFIED: Simplified inventory ranges
        self.inventory_ranges = {
            InventoryZone.GREEN: (30.0, 70.0),
            InventoryZone.YELLOW: (25.0, 75.0),
            InventoryZone.RED: (20.0, 80.0),
            InventoryZone.CRITICAL: (15.0, 85.0)
        }
        
        # Multipliers
        self.zone_multipliers = {
            InventoryZone.GREEN: kwargs.get('green_multiplier', 1.1),
            InventoryZone.YELLOW: kwargs.get('yellow_multiplier', 1.5),
            InventoryZone.RED: kwargs.get('red_multiplier', 2.5),
            InventoryZone.CRITICAL: kwargs.get('critical_multiplier', 5.0)
        }
        
        # Adjustments
        self.base_inv_adjustment = kwargs.get('base_inv_adjustment', 0.0004)
        self.max_inv_adjustment = kwargs.get('max_inv_adjustment', 0.0020)
        
        # Emergency controls
        self.hours_at_extreme_threshold = kwargs.get('hours_at_extreme_threshold', 2.0)
        self.emergency_halt_threshold = kwargs.get('emergency_halt_threshold', 0.06)
        self.halt_duration = kwargs.get('halt_duration', 120)
        
        # Market dynamics
        self.volume_surge_threshold = kwargs.get('volume_surge_threshold', 0.20)
        self.volume_collapse_threshold = kwargs.get('volume_collapse_threshold', -0.20)
        self.price_drop_large = kwargs.get('price_drop_large', 0.005)
        self.price_rise_large = kwargs.get('price_rise_large', 0.005)
        
        # Spread calculations
        self.volatility_add_per_0001 = kwargs.get('volatility_add_per_0001', 0.00030)
        self.market_spread_add_factor = kwargs.get('market_spread_add_factor', 0.6)
        
        # Coin-specific algorithmic parameters
        self.inventory_adjustment_aggression = kwargs.get('inventory_adjustment_aggression', 1.0)
        self.trend_protection_aggression = kwargs.get('trend_protection_aggression', 1.0)
        self.force_rebalance_aggression = kwargs.get('force_rebalance_aggression', 1.0)
        self.spread_asymmetry_factor = kwargs.get('spread_asymmetry_factor', 1.0)
        
        # Regime configurations
        self.regime_configs = kwargs.get('regime_configs', self._default_regime_configs())
        
        # MODIFIED: Add adaptive trade frequency parameters
        self.target_fill_rate_multiplier = kwargs.get('target_fill_rate_multiplier', 1.0)
        self.aggressive_spread_compression = kwargs.get('aggressive_spread_compression', False)
        self.adaptive_refresh_rate = kwargs.get('adaptive_refresh_rate', False)

    def _default_regime_configs(self) -> Dict:
        return {
            MarketRegime.ULTRA_CALM: {
                'base_bid_spread': 0.0027,
                'base_ask_spread': 0.0027,
                'min_spread': 0.0001,
                'position_multiplier': 0.8,
                'target_fill_rate': 1.5,
                'inventory_max_imbalance': 6.0,
                'target_profit_pct': 0.45,
            },
            MarketRegime.CALM: {
                'base_bid_spread': 0.0028,
                'base_ask_spread': 0.0028,
                'min_spread': 0.0001,
                'position_multiplier': 1.0,
                'target_fill_rate': 2.0,
                'inventory_max_imbalance': 7.0,
                'target_profit_pct': 0.40,
            },
            MarketRegime.NORMAL: {
                'base_bid_spread': 0.0030,
                'base_ask_spread': 0.0030,
                'min_spread': 0.0001,
                'position_multiplier': 1.0,
                'target_fill_rate': 2.0,
                'inventory_max_imbalance': 7.0,
                'target_profit_pct': 0.40,
            },
            MarketRegime.VOLATILE: {
                'base_bid_spread': 0.0036,
                'base_ask_spread': 0.0036,
                'min_spread': 0.0001,
                'position_multiplier': 0.7,
                'target_fill_rate': 1.0,
                'inventory_max_imbalance': 5.0,
                'target_profit_pct': 0.55,
            },
            MarketRegime.EXTREME: {
                'base_bid_spread': 0.0048,
                'base_ask_spread': 0.0048,
                'min_spread': 0.0001,
                'position_multiplier': 0.5,
                'target_fill_rate': 0.5,
                'inventory_max_imbalance': 4.0,
                'target_profit_pct': 0.60,
            },
            MarketRegime.TRENDING_UP: {
                'base_bid_spread': 0.0032,
                'base_ask_spread': 0.0042,
                'min_spread': 0.0001,
                'position_multiplier': 0.5,
                'target_fill_rate': 0.5,
                'inventory_max_imbalance': 2.0,
                'target_profit_pct': 0.65,
                'inventory_hard_floor': 48.0,
            },
            MarketRegime.TRENDING_DOWN: {
                'base_bid_spread': 0.0042,
                'base_ask_spread': 0.0032,
                'min_spread': 0.0001,
                'position_multiplier': 0.5,
                'target_fill_rate': 0.5,
                'inventory_max_imbalance': 2.0,
                'target_profit_pct': 0.65,
                'inventory_hard_ceiling': 52.0,
            },
            MarketRegime.CHOPPY: {
                'base_bid_spread': 0.00136,
                'base_ask_spread': 0.00138,
                'min_spread': 0.0001,
                'position_multiplier': 0.8,
                'target_fill_rate': 1.0,
                'inventory_max_imbalance': 6.0,
                'target_profit_pct': 0.50,
            }
        }

# ==================== TECHNICAL ANALYSIS ====================
class TechnicalAnalyzer:
    """
    Analyzes candlestick data from Tickr to provide trading signals
    Works with or without TA-Lib
    """
    
    def __init__(self):
        self.last_analysis = {}
        self.last_analysis_time = 0
        self.analysis_cache_duration = 300  # Cache for 5 minutes
    
    def analyze_candles(self, df, pair_name: str) -> Dict[str, Any]:
        """
        Main analysis function - analyzes candlestick data
        Returns comprehensive technical analysis
        """
        if df is None:
            logger.error(f"{pair_name}: No candle data received from Tickr - df is None")
            return self._get_default_analysis()
        
        if len(df) < 50:
            logger.warning(f"{pair_name}: Insufficient candle data: {len(df)} rows < 50 required")
            return self._get_default_analysis()
        
        # Check cache
        current_time = time.time()
        cache_key = f"{pair_name}_{len(df)}"
        if (current_time - self.last_analysis_time < self.analysis_cache_duration and 
            cache_key in self.last_analysis):
            return self.last_analysis[cache_key]
        
        try:
            analysis = {}
            
            # Ensure we have the right columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"{pair_name}: Missing required columns in candle data")
                return self._get_default_analysis()
            
            df = df.copy()
            # Convert to numeric
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any NaN rows
            df = df.dropna()
            
            if len(df) < 50:
                return self._get_default_analysis()
            
            # 1. Trend Analysis
            analysis['trend'] = self._analyze_trend(df)
            
            # 2. Momentum Analysis
            analysis['momentum'] = self._analyze_momentum(df)
            
            # 3. Volatility Analysis
            analysis['volatility'] = self._analyze_volatility(df)
            
            # 4. Volume Analysis
            analysis['volume'] = self._analyze_volume(df)
            
            # 5. Support/Resistance Levels
            analysis['levels'] = self._find_key_levels(df)
            
            # 6. Candlestick Patterns (if TA-Lib available)
            if TALIB_AVAILABLE:
                analysis['patterns'] = self._detect_patterns(df)
            else:
                analysis['patterns'] = {'signal': 'neutral'}
            
            # 7. Generate Trading Signal
            analysis['signal'] = self._generate_signal(analysis)
            
            # Cache the result
            self.last_analysis[cache_key] = analysis
            self.last_analysis_time = current_time
            
            return analysis
            
        except Exception as e:
            logger.error(f"{pair_name}: Error analyzing candles: {str(e)}")
            return self._get_default_analysis()
    
    def _analyze_trend(self, df) -> Dict:
        """Analyze trend using moving averages"""
        close = df['close'].values
        
        # Calculate EMAs
        ema_fast = self._ema(close, 9)
        ema_medium = self._ema(close, 21)
        ema_slow = self._ema(close, 50)
        
        current_price = close[-1]
        
        # Trend direction
        if ema_fast > ema_medium > ema_slow:
            direction = 'bullish'
            strength = min((ema_fast - ema_slow) / ema_slow * 100, 5.0)
        elif ema_fast < ema_medium < ema_slow:
            direction = 'bearish'
            strength = min((ema_slow - ema_fast) / ema_slow * 100, 5.0)
        else:
            direction = 'neutral'
            strength = 0.0
        
        return {
            'direction': direction,
            'strength': float(strength),
            'ema_9': float(ema_fast),
            'ema_21': float(ema_medium),
            'ema_50': float(ema_slow),
            'above_ema_21': current_price > ema_medium,
            'above_ema_50': current_price > ema_slow
        }
    
    def _analyze_momentum(self, df) -> Dict:
        """Analyze momentum indicators"""
        close = df['close'].values
        
        # RSI calculation
        rsi = self._calculate_rsi(close, 14)
        
        # Rate of Change
        roc = (close[-1] - close[-10]) / close[-10] * 100 if len(close) >= 10 else 0
        
        # Momentum signal
        if rsi > 70:
            signal = 'overbought'
        elif rsi < 30:
            signal = 'oversold'
        else:
            signal = 'neutral'
        
        return {
            'rsi': float(rsi),
            'roc': float(roc),
            'signal': signal,
            'momentum_score': float((rsi - 50) / 50)  # -1 to 1 scale
        }
    
    def _analyze_volatility(self, df) -> Dict:
        """Analyze volatility using Bollinger Bands"""
        close = df['close'].values
        
        # Calculate Bollinger Bands
        sma_20 = np.mean(close[-20:])
        std_20 = np.std(close[-20:])
        
        bb_upper = sma_20 + (2 * std_20)
        bb_lower = sma_20 - (2 * std_20)
        bb_width = (bb_upper - bb_lower) / sma_20 * 100
        
        current_price = close[-1]
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        # ATR approximation
        high = df['high'].values
        low = df['low'].values
        tr = np.maximum(high[-14:] - low[-14:], 
                       np.abs(high[-14:] - close[-15:-1]))
        atr = np.mean(tr)
        atr_pct = atr / current_price * 100
        
        return {
            'bb_upper': float(bb_upper),
            'bb_middle': float(sma_20),
            'bb_lower': float(bb_lower),
            'bb_width_pct': float(bb_width),
            'bb_position': float(bb_position),
            'atr_pct': float(atr_pct),
            'volatility_state': 'high' if bb_width > 4.0 else 'normal' if bb_width > 2.0 else 'low'
        }
    
    def _analyze_volume(self, df) -> Dict:
        """Analyze volume patterns"""
        volume = df['volume'].values
        close = df['close'].values
        
        if len(volume) < 20:
            return {'signal': 'unknown', 'ratio': 1.0}
        
        current_vol = volume[-1]
        avg_vol = np.mean(volume[-20:])
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Volume trend
        vol_trend = 'increasing' if volume[-1] > volume[-5] else 'decreasing'
        
        # Price-Volume relationship
        price_up = close[-1] > close[-2]
        volume_up = volume[-1] > volume[-2]
        
        if price_up and volume_up:
            signal = 'bullish'
        elif not price_up and volume_up:
            signal = 'bearish'
        else:
            signal = 'neutral'
        
        return {
            'current': float(current_vol),
            'average': float(avg_vol),
            'ratio': float(vol_ratio),
            'trend': vol_trend,
            'signal': signal
        }
    
    def _find_key_levels(self, df) -> Dict:
        """Find support and resistance levels"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Recent pivot points
        pivot_high = np.max(high[-20:])
        pivot_low = np.min(low[-20:])
        
        current_price = close[-1]
        
        # Distance to key levels
        dist_to_high = (pivot_high - current_price) / current_price * 100
        dist_to_low = (current_price - pivot_low) / current_price * 100
        
        return {
            'resistance': float(pivot_high),
            'support': float(pivot_low),
            'current_price': float(current_price),
            'dist_to_resistance_pct': float(dist_to_high),
            'dist_to_support_pct': float(dist_to_low),
            'near_resistance': dist_to_high < 1.0,
            'near_support': dist_to_low < 1.0
        }
    
    def _detect_patterns(self, df) -> Dict:
        """Detect candlestick patterns using TA-Lib"""
        if not TALIB_AVAILABLE:
            return {'signal': 'neutral'}
        
        try:
            open_p = df['open'].values
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Check key patterns
            hammer = talib.CDLHAMMER(open_p, high, low, close)[-1] != 0
            shooting_star = talib.CDLSHOOTINGSTAR(open_p, high, low, close)[-1] != 0
            engulfing = talib.CDLENGULFING(open_p, high, low, close)[-1]
            doji = talib.CDLDOJI(open_p, high, low, close)[-1] != 0
            
            bullish_count = int(hammer) + int(engulfing > 0)
            bearish_count = int(shooting_star) + int(engulfing < 0)
            
            if bullish_count > bearish_count:
                signal = 'bullish'
            elif bearish_count > bullish_count:
                signal = 'bearish'
            else:
                signal = 'neutral'
            
            return {
                'signal': signal,
                'hammer': hammer,
                'shooting_star': shooting_star,
                'engulfing': int(engulfing),
                'doji': doji
            }
        except Exception as e:
            logger.debug(f"Pattern detection error: {e}")
            return {'signal': 'neutral'}
    
    def _generate_signal(self, analysis: Dict) -> Dict:
        """Generate overall trading signal from all analyses"""
        score = 0.0
        
        # Trend contribution (35%)
        if analysis['trend']['direction'] == 'bullish':
            score += 0.35 * min(analysis['trend']['strength'] / 2.0, 1.0)
        elif analysis['trend']['direction'] == 'bearish':
            score -= 0.35 * min(analysis['trend']['strength'] / 2.0, 1.0)
        
        # Momentum contribution (30%)
        if analysis['momentum']['signal'] == 'oversold':
            score += 0.30
        elif analysis['momentum']['signal'] == 'overbought':
            score -= 0.30
        else:
            score += 0.30 * analysis['momentum']['momentum_score']
        
        # Volume contribution (20%)
        if analysis['volume']['signal'] == 'bullish':
            score += 0.20
        elif analysis['volume']['signal'] == 'bearish':
            score -= 0.20
        
        # Pattern contribution (15%)
        if 'patterns' in analysis:
            if analysis['patterns']['signal'] == 'bullish':
                score += 0.15
            elif analysis['patterns']['signal'] == 'bearish':
                score -= 0.15
        
        # Normalize to -1 to 1
        score = max(-1.0, min(1.0, score))
        
        # Determine action
        if score >= 0.5:
            action = 'strong_buy'
            confidence = min(score * 100, 100)
        elif score >= 0.2:
            action = 'weak_buy'
            confidence = score * 100
        elif score <= -0.5:
            action = 'strong_sell'
            confidence = min(abs(score) * 100, 100)
        elif score <= -0.2:
            action = 'weak_sell'
            confidence = abs(score) * 100
        else:
            action = 'neutral'
            confidence = 50.0
        
        return {
            'action': action,
            'score': float(score),
            'confidence': float(confidence)
        }
    
    def _ema(self, data, period):
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return data[-1]
        
        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])
        
        for price in data[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _get_default_analysis(self) -> Dict:
        """Return default analysis when data is insufficient"""
        return {
            'trend': {'direction': 'neutral', 'strength': 0.0},
            'momentum': {'rsi': 50.0, 'signal': 'neutral', 'momentum_score': 0.0},
            'volatility': {'bb_width_pct': 2.0, 'volatility_state': 'normal', 'atr_pct': 1.0},
            'volume': {'signal': 'neutral', 'ratio': 1.0},
            'levels': {'near_resistance': False, 'near_support': False},
            'patterns': {'signal': 'neutral'},
            'signal': {'action': 'neutral', 'score': 0.0, 'confidence': 50.0}
        }

# ==================== PRICE DATA ====================
# ==================== PRICE DATA ====================
# REPLACE YOUR EXISTING PriceData CLASS WITH THIS VERSION

class PriceData:
    def __init__(self, max_len=60, pair_name: str = ""):
        self.closes = deque(maxlen=max_len)
        self.volumes = deque(maxlen=max_len)
        self.timestamps = deque(maxlen=max_len)
        self.long_term_trend = 0.0
        self.performance_history = deque(maxlen=300)
        self.market_data = {}
        self.pair_name = pair_name
        
        # NEW: Initialize Tickr and Technical Analyzer
        self.tickr_client = None
        self.technical_analyzer = None
        if TICKR_AVAILABLE:
            try:
                # FIXED: Use Kraken to match paper trading exchange
                self.tickr_client = TickrClient(exchange="kraken")  # Change to "kraken"
                logger.info(f"{self.pair_name}: Tickr client initialized for KRAKEN")
            except Exception as e:
                logger.error(f"{self.pair_name}: Could not initialize Tickr: {e}", exc_info=True)

        
        if TALIB_AVAILABLE:
            self.technical_analyzer = TechnicalAnalyzer()
            logger.info(f"{pair_name}: Technical analyzer initialized")
        
        # NEW: Candle data cache with timeframe-specific durations
        self.candle_cache = {}
        self.last_candle_fetch = {}
        self.candle_cache_durations = {
            '1m': 60,     # 1m
            '5m': 300,    # 5m
            '15m': 900,   # 15m
            '1h': 3600,    # 1h
            '4h': 7200,   # 2h
            '1d': 14400,   # 4h
        }
        
    
    # NEW: Method to fetch candles from Tickr
    def get_tickr_candles(self, timeframe: str = '15m', limit: int = 200, retries: int = 3) -> Optional[Any]:
        """
        Fetch candle data from Tickr with caching + retry logic + exponential backoff
        """
        if not self.tickr_client or not self.pair_name:
            logger.warning(f"Cannot fetch candles: tickr_client or pair_name missing")
            return None

        cache_key = f"{self.pair_name}_{timeframe}_{limit}"
        cache_duration = self.candle_cache_durations.get(timeframe, 300)  # default 5 min
        current_time = time.time()

        # === CACHE CHECK ===
        if (cache_key in self.last_candle_fetch and 
            current_time - self.last_candle_fetch[cache_key] < cache_duration):
            cached_data = self.candle_cache.get(cache_key)
            if cached_data is not None and len(cached_data) > 0:
                logger.debug(f"{self.pair_name}: Using cached {timeframe} candles ({len(cached_data)} bars)")
                return cached_data

        # === FRESH FETCH WITH RETRIES ===
        symbol = f"{self.pair_name}/USDT"
        logger.info(f"{self.pair_name}: Fetching {limit} {timeframe} candles → {symbol}")

        for attempt in range(retries):
            try:
                df = self.tickr_client.get_candles(symbol, timeframe, limit=limit)

                if df is not None and len(df) > 0:
                    # Cache successful result
                    self.candle_cache[cache_key] = df
                    self.last_candle_fetch[cache_key] = current_time

                    first_ts = df.iloc[0]['timestamp'] if 'timestamp' in df.columns else df.iloc[0][0]
                    last_ts = df.iloc[-1]['timestamp'] if 'timestamp' in df.columns else df.iloc[-1][0]

                    logger.info(
                        f"{self.pair_name}: Successfully fetched {len(df)} {timeframe} candles | "
                        f"{datetime.datetime.fromtimestamp(first_ts / 1000).strftime('%Y-%m-%d %H:%M')} → "
                        f"{datetime.datetime.fromtimestamp(last_ts / 1000).strftime('%Y-%m-%d %H:%M')}"
                    )

                    if len(df) < 50:
                        logger.warning(f"{self.pair_name}: Low candle count ({len(df)}), TA may be unreliable")

                    return df

            except Exception as e:
                logger.warning(f"{self.pair_name}: Tickr candle fetch attempt {attempt+1}/{retries} failed: {e}")

                if attempt < retries - 1:
                    sleep_time = 2 ** attempt  # exponential backoff: 1s, 2s, 4s...
                    logger.info(f"{self.pair_name}: Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)

        # === ALL RETRIES FAILED ===
        logger.error(f"{self.pair_name}: Failed to fetch candles from Tickr after {retries} attempts")
        
        # Clear potentially corrupted cache entry
        self.candle_cache.pop(cache_key, None)
        self.last_candle_fetch.pop(cache_key, None)

        return None
    
    
    def get_technical_analysis(self, timeframe: str = '15m') -> Dict:
        """
        Get technical analysis from candlestick data
        """
        if not self.technical_analyzer:
            return {
                'signal': {'action': 'neutral', 'score': 0.0, 'confidence': 50.0},
                'available': False
            }
        
        # Get candle data
        df = self.get_tickr_candles(timeframe=timeframe, limit=200)
        
        if df is None:
            return {
                'signal': {'action': 'neutral', 'score': 0.0, 'confidence': 50.0},
                'available': False
            }
        
        # Analyze
        analysis = self.technical_analyzer.analyze_candles(df, self.pair_name)
        analysis['available'] = True
        analysis['timeframe'] = timeframe
        
        return analysis
    
    def get_multi_timeframe_analysis(self) -> Dict:
        """
        Get analysis from multiple timeframes for better decision making
        """
        if not self.technical_analyzer or not self.tickr_client:
            return {'available': False}
        
        try:
            # Get different timeframes
            short_term = self.get_technical_analysis('15m')  # Short-term (15min)
            medium_term = self.get_technical_analysis('1h')   # Medium-term (1 hour)
            long_term = self.get_technical_analysis('4h')     # Long-term (4 hours)
            
            # Check data quality and apply penalties
            data_quality_multiplier = 1.0
            total_candles = 0
            available_timeframes = 0
            
            for tf_name, tf_data in [('short', short_term), ('medium', medium_term), ('long', long_term)]:
                if tf_data.get('available'):
                    available_timeframes += 1
                    # Check if we have the candle data
                    tf_label = {'short': '15m', 'medium': '1h', 'long': '4h'}[tf_name]
                    cache_key = f"{self.pair_name}_{tf_label}_200"
                    if cache_key in self.candle_cache:
                        candle_count = len(self.candle_cache[cache_key])
                        total_candles += candle_count
                        if candle_count < 100:
                            data_quality_multiplier *= 0.9  # 10% penalty per low-quality timeframe
            
            if available_timeframes < 2:
                logger.warning(f"{self.pair_name}: Insufficient timeframes available ({available_timeframes}/3)")
                data_quality_multiplier *= 0.7  # 30% penalty for missing timeframes
            
            # Combine signals
            scores = []
            if short_term.get('available'):
                scores.append(short_term['signal']['score'] * 0.3)  # 30% weight
            if medium_term.get('available'):
                scores.append(medium_term['signal']['score'] * 0.4)  # 40% weight
            if long_term.get('available'):
                scores.append(long_term['signal']['score'] * 0.3)   # 30% weight
            
            combined_score = sum(scores) if scores else 0.0
            
            # Volume confirmation bonus
            volume_confirms = False
            if medium_term.get('available'):
                vol_signal = medium_term.get('volume', {}).get('signal', 'neutral')
                trend_direction = medium_term.get('trend', {}).get('direction', 'neutral')
                
                if (vol_signal == 'bullish' and trend_direction == 'bullish') or \
                   (vol_signal == 'bearish' and trend_direction == 'bearish'):
                    volume_confirms = True
                    combined_score *= 1.1  # 10% boost for volume confirmation
                    logger.info(f"{self.pair_name}: Volume confirms trend direction")
            
            # Determine combined action
            if combined_score >= 0.5:
                action = 'strong_buy'
            elif combined_score >= 0.2:
                action = 'weak_buy'
            elif combined_score <= -0.5:
                action = 'strong_sell'
            elif combined_score <= -0.2:
                action = 'weak_sell'
            else:
                action = 'neutral'
            
            # Detect conflicts between timeframes
            conflict_detected = False
            if short_term.get('available') and long_term.get('available'):
                short_action = short_term['signal']['action']
                long_action = long_term['signal']['action']
                
                # Check for major conflicts (buy vs sell)
                if ('buy' in short_action and 'sell' in long_action) or \
                   ('sell' in short_action and 'buy' in long_action):
                    conflict_detected = True
                    logger.warning(f"{self.pair_name}: Conflicting signals detected - "
                                f"Short-term: {short_action}, Long-term: {long_action}")
                    # Reduce confidence when conflicted
                    combined_confidence = abs(combined_score) * 50  # Halve the confidence
                else:
                    combined_confidence = abs(combined_score) * 100
            else:
                combined_confidence = abs(combined_score) * 100
            
            # Apply data quality penalty to confidence
            combined_confidence *= data_quality_multiplier
            
            # Check staleness of data
            staleness_penalty = 1.0
            if long_term.get('available'):
                cache_key = f"{self.pair_name}_4h_200"
                if cache_key in self.last_candle_fetch:
                    staleness = time.time() - self.last_candle_fetch[cache_key]
                    if staleness > 7200:  # 2 hours old
                        staleness_hours = staleness / 3600
                        logger.warning(f"{self.pair_name}: Candle data stale ({staleness_hours:.1f}h old)")
                        staleness_penalty = 0.5
                        combined_confidence *= staleness_penalty
            
            return {
                'available': True,
                'short_term': short_term,
                'medium_term': medium_term,
                'long_term': long_term,
                'combined_signal': {
                    'action': action,
                    'score': combined_score,
                    'confidence': combined_confidence,
                    'conflict_detected': conflict_detected,
                    'volume_confirms': volume_confirms,
                    'data_quality': data_quality_multiplier,
                    'staleness_penalty': staleness_penalty,
                    'available_timeframes': available_timeframes
                }
            }
        except Exception as e:
            logger.error(f"{self.pair_name}: Error in multi-timeframe analysis: {e}", exc_info=True)
            return {'available': False}
    
    def get_coin_market_data(self, coin: str) -> Optional[Dict]:
        """Get market data for a specific coin"""
        return self.market_data.get(coin)
    
    def add_price(self, close: float, volume: float = 0, timestamp: Optional[float] = None):
        self.closes.append(close)
        self.volumes.append(volume)
        self.timestamps.append(timestamp or time.time())
    
    def get_recent(self, window: int) -> Tuple[list, list]:
        closes = list(self.closes)[-window:] if self.closes else []
        volumes = list(self.volumes)[-window:] if self.volumes else []
        return closes, volumes
    
    def add_performance_data(self, fill_rate: float, profit: float, timestamp: float):
        self.performance_history.append({
            'timestamp': timestamp,
            'fill_rate': fill_rate,
            'profit': profit
        })
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        if len(self.performance_history) < 10:
            return {'status': 'insufficient_data'}
        
        recent_data = list(self.performance_history)[-60:]
        if len(recent_data) < 10:
            return {'status': 'insufficient_data'}
        
        avg_fill_rate = statistics.mean([d['fill_rate'] for d in recent_data])
        avg_profit = statistics.mean([d['profit'] for d in recent_data])
        
        if len(recent_data) >= 20:
            first_half = recent_data[:len(recent_data)//2]
            second_half = recent_data[len(recent_data)//2:]
            first_avg = statistics.mean([d['fill_rate'] for d in first_half])
            second_avg = statistics.mean([d['fill_rate'] for d in second_half])
            fill_rate_trend = second_avg - first_avg
        else:
            fill_rate_trend = 0.0
        
        return {
            'status': 'ok',
            'avg_fill_rate': avg_fill_rate,
            'avg_profit': avg_profit,
            'fill_rate_trend': fill_rate_trend,
            'data_points': len(recent_data)
        }
# ==================== OPTIMIZER ====================
class OptimizedOptimizer:
    def __init__(self, config: BaseConfig, pair_name: str):
        self.config = config
        self.pair_name = pair_name
        self.price_data = PriceData(pair_name=pair_name)  # MODIFIED: Pass pair_name
        self.halt_until = 0
        self.time_at_extreme_start = None
        self.last_fill_time = time.time()
        self.fill_intervals = deque(maxlen=10)
        self.long_term_trend = 0.0
        self.last_api_call_time = 0
        self.api_call_interval = 5
        
        self.start_time = time.time()
        self.total_fills = 0
        self.total_trades = 0
        self.performance_checks = 0
        self.adaptive_adjustments = 0
        
        self.aggression_level = 1.0
        self.spread_compression_factor = 1.0
        self.target_fill_rate_boost = 1.0

        # Log Tickr integration status
        if self.price_data.tickr_client:
            logger.info(f"{pair_name}: Tickr integration ACTIVE - Technical analysis enabled")
        else:
            logger.warning(f"{pair_name}: Tickr integration INACTIVE - Using price-based analysis only")

        if self.price_data.technical_analyzer:
            logger.info(f"{pair_name}: Technical analyzer available: "
                    f"TA-Lib={'available' if TALIB_AVAILABLE else 'not available'}")
        
    
    def update_live_market_data(self):
        """Update market data based on live price history"""
        prices, _ = self.price_data.get_recent(60)
        
        if len(prices) >= 20:
            current_price = prices[-1]
            avg_price = statistics.mean(prices)
            volatility = statistics.stdev(prices) / avg_price if avg_price > 0 else 0.001
            
            # Calculate trend from available data
            if len(prices) >= 40:
                first_half = prices[:20]
                second_half = prices[-20:]
                trend = (statistics.mean(second_half) - statistics.mean(first_half)) / statistics.mean(first_half)
            else:
                trend = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0.0
            
            # Update market data
            self.price_data.market_data[self.pair_name] = {
                'current_price': current_price,
                'avg_price': avg_price,
                'volatility': volatility,
                'trend': trend,
                'data_points': len(prices)
            }
            
            # Update long-term trend
            self.long_term_trend = trend
            
    def add_price(self, price: float, volume: float = 0):
        self.price_data.add_price(price, volume, time.time())
    
    def track_fill(self):
        current_time = time.time()
        if self.last_fill_time > 0:
            self.fill_intervals.append(current_time - self.last_fill_time)
        self.last_fill_time = current_time
        self.total_fills += 1
        
        # MODIFIED: Track performance for adaptive adjustments
        self.price_data.add_performance_data(
            self.get_current_fill_rate(),
            0.0,
            current_time
        )
    
    def get_current_fill_rate(self) -> float:
        if len(self.fill_intervals) < 2:
            return 0.0
        avg_interval = statistics.mean(self.fill_intervals)
        return 3600.0 / avg_interval if avg_interval > 0 else 0.0
    
    def get_inventory_zone(self, inventory_pct: float) -> InventoryZone:
        for zone, (min_val, max_val) in self.config.inventory_ranges.items():
            if min_val <= inventory_pct <= max_val:
                return zone
        # Default to critical if outside all ranges
        return InventoryZone.CRITICAL
    
    def calculate_volatility(self, prices: list) -> float:
        if len(prices) < 3:
            return 0.0
        
        try:
            returns = [abs((prices[i] - prices[i-1]) / prices[i-1]) 
                      for i in range(1, len(prices)) if prices[i-1] != 0]
            
            if not returns:
                return 0.0
            
            window = min(self.config.volatility_window, len(returns))
            recent = returns[-window:]
            
            mean_vol = statistics.mean(recent)
            if len(recent) >= 2:
                stdev_vol = statistics.stdev(recent)
                return (mean_vol + stdev_vol) / 2
            return mean_vol
        except:
            return 0.0
    
    def calculate_momentum(self, prices: list) -> float:
        if len(prices) < self.config.momentum_window + 1:
            return 0.0
        current = prices[-1]
        past = prices[-(self.config.momentum_window + 1)]
        return (current - past) / past if past != 0 else 0.0
    
    def calculate_trend(self, prices: list) -> Tuple[float, float, float, float]:
        if len(prices) < self.config.slow_ma_window:
            current = prices[-1] if prices else 0.0
            return current, current, 0.0, 0.0
        
        try:
            fast_ma = statistics.mean(prices[-self.config.fast_ma_window:])
            slow_ma = statistics.mean(prices[-self.config.slow_ma_window:])
            current = prices[-1]
            price_dev = (current - slow_ma) / slow_ma if slow_ma != 0 else 0.0
            ma_dev = (fast_ma - slow_ma) / slow_ma if slow_ma != 0 else 0.0
            return fast_ma, slow_ma, price_dev, ma_dev
        except:
            current = prices[-1] if prices else 0.0
            return current, current, 0.0, 0.0
    
    def detect_regime(self, volatility: float, trend_dev: float, momentum: float, 
                     ma_cross: float, price_range_pct: float) -> MarketRegime:
        config = self.config
        
        is_large_move = price_range_pct > config.price_drop_large
        
        if volatility > config.volatility_extreme_threshold or is_large_move:
            return MarketRegime.EXTREME
        elif (abs(trend_dev) > config.trend_strong_threshold and 
              abs(momentum) > config.momentum_strong_threshold and
              abs(ma_cross) > 0.0003):
            # MODIFIED: Incorporate long-term trend for regime detection
            if (trend_dev > 0 and momentum > 0) or self.long_term_trend > 0:
                return MarketRegime.TRENDING_UP
            elif (trend_dev < 0 and momentum < 0) or self.long_term_trend < 0:
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.CHOPPY
        elif volatility > config.volatility_high_threshold:
            return MarketRegime.VOLATILE
        elif volatility > config.volatility_low_threshold:
            return MarketRegime.CHOPPY if abs(ma_cross) >= 0.0002 else MarketRegime.NORMAL
        elif volatility < config.volatility_ultra_low_threshold and price_range_pct < 0.002:
            return MarketRegime.ULTRA_CALM
        elif volatility < config.volatility_low_threshold:
            return MarketRegime.CALM
        else:
            return MarketRegime.NORMAL

    def calculate_spreads(self, regime: MarketRegime, volatility: float, trend_dev: float,
                         momentum: float, inventory_pct: float, market_spread: float,
                         volume_regime: str, volume_change: float) -> Tuple[float, float]:
        regime_cfg = self.config.regime_configs.get(regime, self.config.regime_configs[MarketRegime.NORMAL])
        bid_spread = regime_cfg['base_bid_spread']
        ask_spread = regime_cfg['base_ask_spread']
        min_spread = regime_cfg.get('min_spread', self.config.absolute_min_spread)
        
        # MODIFIED: Reduced minimum spreads to improve fill rates
        min_spread = max(0.00035, min_spread)
        
        # MODIFIED: Apply adaptive spread compression based on performance
        if self.config.aggressive_spread_compression:
            bid_spread *= self.spread_compression_factor
            ask_spread *= self.spread_compression_factor
        
        # Volatility adjustment
        vol_add = min((volatility / 0.0001) * self.config.volatility_add_per_0001, 0.0020)
        bid_spread += vol_add
        ask_spread += vol_add
        
        # Market spread buffer
        if market_spread > 0:
            market_add = min(market_spread * self.config.market_spread_add_factor, 0.0008)
            bid_spread += market_add
            ask_spread += market_add
        
        # Volume adjustments
        if volume_regime == 'surge':
            surge_add = min(abs(volume_change) * 0.015, 0.0008)
            bid_spread += surge_add
            ask_spread += surge_add
        elif volume_regime == 'collapse':
            collapse_reduce = min(abs(volume_change) * 0.008, 0.0003)
            bid_spread = max(bid_spread - collapse_reduce, min_spread)
            ask_spread = max(ask_spread - collapse_reduce, min_spread)
        
        # Inventory adjustment with coin-specific aggression
        imbalance = inventory_pct - self.config.inventory_target_pct
        abs_imb = abs(imbalance)
        max_imbalance = regime_cfg.get('inventory_max_imbalance', 7.0)
        zone = self.get_inventory_zone(inventory_pct)

        multiplier = self.config.zone_multipliers[zone]

        # MODIFIED: Apply adaptive aggression
        multiplier *= self.aggression_level

        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            multiplier *= 1.3 * self.config.trend_protection_aggression  # Reduced from 1.5

        if abs_imb > max_imbalance:
            base_adj = self.config.base_inv_adjustment * ((abs_imb / max_imbalance) ** 1.8)  # Reduced exponent from 2.0
        else:
            base_adj = self.config.base_inv_adjustment * (abs_imb / max_imbalance)

        base_adj = min(base_adj * multiplier * self.config.inventory_adjustment_aggression, 
                    self.config.max_inv_adjustment * self.config.inventory_adjustment_aggression)

        if imbalance > 0:
            bid_adj = base_adj * 1.5  # Reduced from 1.8
            ask_adj = -base_adj * 0.7  # Reduced from 0.9 (more aggressive sell reduction)
        else:
            bid_adj = -base_adj * 0.7  # Reduced from 0.9 (more aggressive buy reduction)
            ask_adj = base_adj * 1.5  # Reduced from 1.8

        bid_spread += bid_adj
        ask_spread += ask_adj
        
        # Emergency trend protection with coin-specific aggression
        # Emergency trend protection with coin-specific aggression
        if regime == MarketRegime.TRENDING_DOWN and inventory_pct > 53.0:  # Changed from 51.0
            emergency_factor = (inventory_pct - 50.0) / 10.0
            bid_spread += min(emergency_factor * 0.0006 * self.config.trend_protection_aggression, 0.0012)  # Reduced
            ask_spread -= min(emergency_factor * 0.0003 * self.config.trend_protection_aggression, 0.0006)  # Reduced
            if inventory_pct > 60.0:  # Changed from 55.0
                bid_spread = self.config.absolute_max_spread
                ask_spread = max(min_spread, ask_spread * 0.85)  # Changed from 0.8
        elif regime == MarketRegime.TRENDING_UP and inventory_pct < 47.0:  # Changed from 49.0
            emergency_factor = (50.0 - inventory_pct) / 10.0
            ask_spread += min(emergency_factor * 0.0006 * self.config.trend_protection_aggression, 0.0012)  # Reduced
            bid_spread -= min(emergency_factor * 0.0003 * self.config.trend_protection_aggression, 0.0006)  # Reduced
            if inventory_pct < 40.0:  # Changed from 45.0
                ask_spread = self.config.absolute_max_spread
                bid_spread = max(min_spread, bid_spread * 0.85)  # Changed from 0.8
        
        # Fill rate adjustment with adaptive target
        current_fill_rate = self.get_current_fill_rate()
        target_fill_rate = regime_cfg.get('target_fill_rate', 2.0) * self.target_fill_rate_boost
        if current_fill_rate > target_fill_rate * 1.15:
            fill_rate_penalty = 0.0004 * (current_fill_rate / target_fill_rate - 1.0)
            fill_rate_penalty = min(fill_rate_penalty, 0.0010)
            bid_spread += fill_rate_penalty
            ask_spread += fill_rate_penalty
        
        # Apply bounds
        bid_spread = max(min_spread, min(self.config.absolute_max_spread, bid_spread))
        ask_spread = max(min_spread, min(self.config.absolute_max_spread, ask_spread))
        
        # Fee coverage enforcement
        avg_spread = (bid_spread + ask_spread) / 2
        fee_coverage = (avg_spread * 2) / 0.0052  # Kraken round-trip fee
        if fee_coverage < self.config.minimum_fee_coverage:
            scale_factor = self.config.minimum_fee_coverage / fee_coverage
            bid_spread *= scale_factor
            ask_spread *= scale_factor
            bid_spread = max(min_spread, min(self.config.absolute_max_spread, bid_spread))
            ask_spread = max(min_spread, min(self.config.absolute_max_spread, ask_spread))
        
        # Ensure spread asymmetry with coin-specific factor
        spread_diff = abs(bid_spread - ask_spread)
        if spread_diff < (0.00008 / self.config.spread_asymmetry_factor):
            if imbalance >= 0:
                ask_spread *= (0.96 / self.config.spread_asymmetry_factor)
                bid_spread *= (1.04 * self.config.spread_asymmetry_factor)
            else:
                bid_spread *= (0.96 / self.config.spread_asymmetry_factor)
                ask_spread *= (1.04 * self.config.spread_asymmetry_factor)
        
        return bid_spread, ask_spread
    
    # MODIFIED: Add method to analyze performance and adjust parameters
    def analyze_and_adjust_performance(self):
        """Analyze 5-hour performance and adjust trading parameters for increased frequency"""
        current_time = time.time()
        runtime_hours = (current_time - self.start_time) / 3600.0
        
        # Only adjust after 3 hours of runtime
        if runtime_hours < 3.0:
            return
            
        self.performance_checks += 1
        
        # Get performance analysis
        perf_analysis = self.price_data.get_performance_analysis()
        if perf_analysis['status'] != 'ok':
            return
            
        current_fill_rate = perf_analysis['avg_fill_rate']
        fill_rate_trend = perf_analysis['fill_rate_trend']
        
        # Target fill rates based on market data analysis
        target_fill_rates = {
            'ATOM': 2.0,
            'AVAX': 3.0,
            'DOT': 3.5,  
            'ADA': 4.5,
            'LTC': 4.0,   
        }
        
        target_fill_rate = target_fill_rates.get(self.pair_name, 3.0)
        
        # Calculate adjustment factors
        fill_rate_ratio = current_fill_rate / target_fill_rate if target_fill_rate > 0 else 1.0
        
        # If fill rate is below target, increase aggression
        if fill_rate_ratio < 0.8 or fill_rate_trend < 0:
            # Increase aggression level
            self.aggression_level = min(self.aggression_level * 1.15, 2.0)
            
            # Increase target fill rate boost
            self.target_fill_rate_boost = min(self.target_fill_rate_boost * 1.2, 2.0)
            
            # Apply spread compression
            if self.config.aggressive_spread_compression:
                self.spread_compression_factor = max(self.spread_compression_factor * 0.9, 0.7)
            
            self.adaptive_adjustments += 1
            logger.info(f"{self.pair_name}: Performance adjustment #{self.adaptive_adjustments} - "
                       f"Aggression: {self.aggression_level:.2f}, "
                       f"Target boost: {self.target_fill_rate_boost:.2f}")
        
        # If fill rate is above target, reduce aggression slightly
        elif fill_rate_ratio > 1.2 and fill_rate_trend > 0:
            self.aggression_level = max(self.aggression_level * 0.95, 1.0)
            self.target_fill_rate_boost = max(self.target_fill_rate_boost * 0.95, 1.0)
            if self.config.aggressive_spread_compression:
                self.spread_compression_factor = min(self.spread_compression_factor * 1.05, 1.0)
    
    def get_tickr_adjustments(self) -> Dict[str, Any]:
        """
        Get spread and position adjustments based on Tickr technical analysis
        Returns adjustments to apply to base spreads
        """
        if not TICKR_AVAILABLE or not self.price_data.tickr_client:
            logger.debug(f"{self.pair_name}: Tickr adjustments skipped - Tickr not available")
            return {
                'bid_adjustment': 0.0,
                'ask_adjustment': 0.0,
                'position_mult': 1.0,
                'confidence': 0.0,
                'reason': 'Tickr not available'
            }
        
        # Verify technical analyzer is also available
        if not self.price_data.technical_analyzer:
            logger.debug(f"{self.pair_name}: Tickr adjustments skipped - Technical analyzer not initialized")
            return {
                'bid_adjustment': 0.0,
                'ask_adjustment': 0.0,
                'position_mult': 1.0,
                'confidence': 0.0,
                'reason': 'Technical analyzer not available'
            }
        
        try:
            # Get multi-timeframe analysis
            mtf_analysis = self.price_data.get_multi_timeframe_analysis()
            
            if not mtf_analysis.get('available'):
                return {
                    'bid_adjustment': 0.0,
                    'ask_adjustment': 0.0,
                    'position_mult': 1.0,
                    'confidence': 0.0,
                    'reason': 'No candle data available'
                }
            
            # Get the combined signal
            signal = mtf_analysis['combined_signal']
            action = signal['action']
            score = signal['score']
            confidence = signal['confidence']
            
            # Log data quality metrics
            if signal.get('data_quality', 1.0) < 0.9:
                logger.info(f"{self.pair_name}: Reduced signal quality: {signal['data_quality']:.2f}")
            
            if signal.get('conflict_detected', False):
                logger.warning(f"{self.pair_name}: Timeframe conflict detected in signal")
            
            # Get medium-term analysis for detailed adjustments
            medium = mtf_analysis.get('medium_term', {})
            
            bid_adjustment = 0.0
            ask_adjustment = 0.0
            position_mult = 1.0
            reason = f"{action} signal (confidence: {confidence:.0f}%)"
            
            # Adjust based on signal
            if action == 'strong_buy':
                # Tighten ask spread to capture upside, widen bid for protection
                ask_adjustment = -0.0003  # Reduce ask spread by 0.03%
                bid_adjustment = 0.0002   # Increase bid spread by 0.02%
                position_mult = 1.15      # Increase position size
                
            elif action == 'weak_buy':
                ask_adjustment = -0.0002
                bid_adjustment = 0.0001
                position_mult = 1.08
                
            elif action == 'strong_sell':
                # Tighten bid spread to exit faster, widen ask for protection
                bid_adjustment = -0.0003
                ask_adjustment = 0.0002
                position_mult = 0.85
                
            elif action == 'weak_sell':
                bid_adjustment = -0.0002
                ask_adjustment = 0.0001
                position_mult = 0.92
            
            # Additional adjustments based on technical factors
            if medium.get('available'):
                # Volatility adjustment
                volatility = medium.get('volatility', {})
                if volatility.get('volatility_state') == 'high':
                    # Widen spreads in high volatility
                    bid_adjustment += 0.0002
                    ask_adjustment += 0.0002
                    position_mult *= 0.9
                    reason += " | High volatility"
                elif volatility.get('volatility_state') == 'low':
                    # Tighten spreads in low volatility
                    bid_adjustment -= 0.0001
                    ask_adjustment -= 0.0001
                    position_mult *= 1.05
                    reason += " | Low volatility"
                
                # Support/Resistance adjustment
                levels = medium.get('levels', {})
                if levels.get('near_resistance'):
                    # Widen ask spread near resistance
                    ask_adjustment += 0.0002
                    reason += " | Near resistance"
                if levels.get('near_support'):
                    # Widen bid spread near support
                    bid_adjustment += 0.0002
                    reason += " | Near support"
                
                # Trend adjustment
                trend = medium.get('trend', {})
                if trend.get('direction') == 'bullish' and trend.get('strength', 0) > 2.0:
                    # Strong uptrend - bias toward buying
                    ask_adjustment -= 0.0001
                    position_mult *= 1.05
                    reason += " | Strong uptrend"
                elif trend.get('direction') == 'bearish' and trend.get('strength', 0) > 2.0:
                    # Strong downtrend - bias toward selling
                    bid_adjustment -= 0.0001
                    position_mult *= 1.05
                    reason += " | Strong downtrend"
                
                # Volume confirmation adjustment
                if signal.get('volume_confirms', False):
                    # Boost adjustments slightly when volume confirms
                    if action in ['strong_buy', 'weak_buy']:
                        ask_adjustment -= 0.0001  # Even tighter ask
                    elif action in ['strong_sell', 'weak_sell']:
                        bid_adjustment -= 0.0001  # Even tighter bid
                    reason += " | Volume confirms"
            
            # Reduce adjustments if conflict detected
            if signal.get('conflict_detected', False):
                bid_adjustment *= 0.5
                ask_adjustment *= 0.5
                position_mult = 0.5 * position_mult + 0.5 * 1.0  # Move halfway back to 1.0
                reason += " | CONFLICT: reduced adjustments"
            
            # Reduce adjustments based on data quality
            quality_factor = signal.get('data_quality', 1.0)
            if quality_factor < 1.0:
                bid_adjustment *= quality_factor
                ask_adjustment *= quality_factor
                position_mult = quality_factor * position_mult + (1 - quality_factor) * 1.0
            
            # Cap adjustments to reasonable ranges (increased for more impact)
            bid_adjustment = max(-0.0015, min(0.0015, bid_adjustment))
            ask_adjustment = max(-0.0015, min(0.0015, ask_adjustment))
            position_mult = max(0.7, min(1.3, position_mult))
            
            logger.info(f"{self.pair_name}: Tickr adjustments - "
                    f"Bid: {bid_adjustment*100:+.3f}%, Ask: {ask_adjustment*100:+.3f}%, "
                    f"Pos: {position_mult:.2f}x | {reason}")
            
            return {
                'bid_adjustment': float(bid_adjustment),
                'ask_adjustment': float(ask_adjustment),
                'position_mult': float(position_mult),
                'confidence': float(confidence),
                'reason': reason,
                'signal_action': action,
                'signal_score': float(score),
                'data_quality': float(quality_factor),
                'conflict_detected': signal.get('conflict_detected', False),
                'volume_confirms': signal.get('volume_confirms', False)
            }
            
        except Exception as e:
            logger.error(f"{self.pair_name}: Error getting Tickr adjustments: {str(e)}", exc_info=True)
            return {
                'bid_adjustment': 0.0,
                'ask_adjustment': 0.0,
                'position_mult': 1.0,
                'confidence': 0.0,
                'reason': f'Error: {str(e)}'
            }

    def check_tickr_health(self) -> bool:
        """
        Periodically check if Tickr integration is working
        Returns True if healthy, False otherwise
        """
        if not self.price_data.tickr_client:
            return False
        
        try:
            # Try to fetch a small sample of candles
            df = self.price_data.get_tickr_candles(timeframe='15m', limit=10)
            
            if df is None or len(df) == 0:
                logger.error(f"{self.pair_name}: Tickr health check FAILED - No data returned")
                return False
            
            logger.info(f"{self.pair_name}: Tickr health check PASSED - {len(df)} candles retrieved")
            return True
            
        except Exception as e:
            logger.error(f"{self.pair_name}: Tickr health check FAILED - {str(e)}", exc_info=True)
            return False

    def optimize(self, inventory_pct: float, market_spread: float) -> Dict[str, Any]:
        prices, volumes = self.price_data.get_recent(60)
        
        if len(prices) < 10:
            return self._get_defaults(len(prices))
        
        # UPDATE: Add live market data accumulation
        if len(prices) >= 20:
            self.update_live_market_data()
        
        # MODIFIED: Check API call interval to reduce excessive calls
        current_time = time.time()
        if current_time - self.last_api_call_time < self.api_call_interval:
            pass
        self.last_api_call_time = current_time
        
        # MODIFIED: Perform adaptive performance analysis
        self.analyze_and_adjust_performance()
        
        # Emergency halt check
        if self.halt_until > time.time():
            return self._get_halt_response(f"HALTED for {int(self.halt_until - time.time())}s")
        
        if len(prices) >= 12:
            hour_ago_price = prices[-12]
            current_price = prices[-1]
            move = abs(current_price - hour_ago_price) / hour_ago_price
            if move > self.config.emergency_halt_threshold:
                self.halt_until = time.time() + self.config.halt_duration
                return self._get_halt_response(f"EMERGENCY: {move*100:.2f}% move")
        
        # Calculate metrics
        current_price = prices[-1]
        volatility = self.calculate_volatility(prices)
        momentum = self.calculate_momentum(prices)
        fast_ma, slow_ma, trend_dev, ma_dev = self.calculate_trend(prices)
        
        # Price range
        window = min(20, len(prices))
        recent = prices[-window:]
        price_range_pct = (max(recent) - min(recent)) / min(recent) if min(recent) > 0 else 0.0
        
        # Volume regime
        if len(volumes) >= 2:
            current_vol = volumes[-1]
            previous_vol = volumes[-2]
            if previous_vol != 0 and current_vol != 0:
                volume_change = (current_vol - previous_vol) / previous_vol
                if volume_change > self.config.volume_surge_threshold:
                    volume_regime = 'surge'
                elif volume_change < self.config.volume_collapse_threshold:
                    volume_regime = 'collapse'
                else:
                    volume_regime = 'normal'
            else:
                volume_regime = 'normal'
                volume_change = 0.0
        else:
            volume_regime = 'normal'
            volume_change = 0.0
        
        # Detect regime
        regime = self.detect_regime(volatility, trend_dev, momentum, ma_dev, price_range_pct)
        
        # Inventory halt check
        if regime == MarketRegime.TRENDING_DOWN:
            if inventory_pct > 85.0:
                return self._get_halt_response(f"TRENDING_DOWN: Inventory {inventory_pct:.1f}% > 85%")
        elif regime == MarketRegime.TRENDING_UP:
            if inventory_pct < 15.0:
                return self._get_halt_response(f"TRENDING_UP: Inventory {inventory_pct:.1f}% < 15%")
        
        if inventory_pct > 95.0:
            return self._get_halt_response(f"CRITICAL HIGH: {inventory_pct:.1f}%")
        if inventory_pct < 5.0:
            return self._get_halt_response(f"CRITICAL LOW: {inventory_pct:.1f}%")
        
        # Calculate base spreads
        bid_spread, ask_spread = self.calculate_spreads(
            regime, volatility, trend_dev, momentum, inventory_pct,
            market_spread, volume_regime, volume_change
        )
        
        # NEW: Get Tickr-based adjustments
        tickr_adj = self.get_tickr_adjustments()

        # NEW: Apply Tickr adjustments to spreads with enhanced validation
        if tickr_adj['confidence'] > 65.0:  # Increased threshold from 50% to 65%
            # Store original spreads for potential revert
            original_bid = bid_spread
            original_ask = ask_spread
            
            # Log adjustment attempt
            logger.debug(f"{self.pair_name}: Attempting Tickr adjustments - "
                        f"Bid: {tickr_adj['bid_adjustment']*100:+.3f}%, "
                        f"Ask: {tickr_adj['ask_adjustment']*100:+.3f}%, "
                        f"Confidence: {tickr_adj['confidence']:.1f}%")
            
            bid_spread += Decimal(str(tickr_adj['bid_adjustment']))
            ask_spread += Decimal(str(tickr_adj['ask_adjustment']))
            
            # Ensure spreads stay within bounds after adjustment
            bid_spread = max(Decimal("0.00035"), min(self.config.absolute_max_spread, bid_spread))
            ask_spread = max(Decimal("0.00035"), min(self.config.absolute_max_spread, ask_spread))
            
            # Validate fee coverage after Tickr adjustments
            avg_spread_after = (bid_spread + ask_spread) / 2
            fee_coverage_after = (avg_spread_after * 2) / Decimal("0.0052")
            
            if fee_coverage_after < self.config.minimum_fee_coverage:
                logger.warning(f"{self.pair_name}: Tickr adjustments violated fee coverage "
                            f"({float(fee_coverage_after):.2f}x < {self.config.minimum_fee_coverage}x). "
                            f"Reverting adjustments.")
                bid_spread = original_bid
                ask_spread = original_ask
            else:
                logger.info(f"{self.pair_name}: Applied Tickr adjustments - "
                        f"Bid: {float(bid_spread)*100:.3f}%, Ask: {float(ask_spread)*100:.3f}%, "
                        f"Fee coverage: {float(fee_coverage_after):.2f}x")
                
                # Log additional context
                if tickr_adj.get('conflict_detected'):
                    logger.info(f"{self.pair_name}: Note: Adjustments reduced due to timeframe conflict")
                if tickr_adj.get('volume_confirms'):
                    logger.info(f"{self.pair_name}: Volume confirmation boosted signal strength")
                if tickr_adj.get('data_quality', 1.0) < 0.9:
                    logger.info(f"{self.pair_name}: Adjustments scaled by data quality: "
                            f"{tickr_adj['data_quality']:.2f}")
        else:
            logger.debug(f"{self.pair_name}: Tickr adjustments skipped - "
                        f"Confidence {tickr_adj['confidence']:.1f}% below 65% threshold")
        
        # Forced rebalance check
        is_extreme = inventory_pct <= 20.0 or inventory_pct >= 80.0
        if is_extreme:
            if self.time_at_extreme_start is None:
                self.time_at_extreme_start = time.time()
            hours_at_extreme = (time.time() - self.time_at_extreme_start) / 3600.0
        else:
            self.time_at_extreme_start = None
            hours_at_extreme = 0.0
        
        threshold = self.config.hours_at_extreme_threshold
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            threshold = 1.0
        
        force_action = None
        force_amount = 0.0
        force_price_adj = 0.0
        
        if hours_at_extreme >= threshold:
            if inventory_pct > 80.0:
                force_action = 'force_sell'
                force_amount = 0.01 * self.config.force_rebalance_aggression
                force_price_adj = -0.0005
            elif inventory_pct < 20.0:
                force_action = 'force_buy'
                force_amount = 0.01 * self.config.force_rebalance_aggression
                force_price_adj = 0.0005
        
        # Calculate metrics
        avg_spread = (bid_spread + ask_spread) / 2
        expected_profit_pct = avg_spread * 2 * 100
        fee_coverage = (avg_spread * 2) / 0.0052
        
        # Fee status
        if fee_coverage < 1.0:
            fee_status = 'CRITICAL'
        elif fee_coverage < 1.15:
            fee_status = 'WARNING'
        elif fee_coverage < 1.3:
            fee_status = 'OK'
        else:
            fee_status = 'GOOD'
        
        # Trend direction
        trend = 'neutral'
        if abs(trend_dev) > 0.0003:
            trend = 'bullish' if trend_dev > 0 else 'bearish'
        
        # Inventory status
        inv_zone = self.get_inventory_zone(inventory_pct)
        inv_imbalance = inventory_pct - self.config.inventory_target_pct
        
        inv_status_map = {
            InventoryZone.GREEN: 'OK',
            InventoryZone.YELLOW: 'WARNING',
            InventoryZone.RED: 'CRITICAL',
            InventoryZone.CRITICAL: 'EMERGENCY'
        }
        inv_status = inv_status_map[inv_zone]
        
        # Fill rate
        current_fill_rate = self.get_current_fill_rate()
        target_fill_rate = self.config.regime_configs[regime].get('target_fill_rate', 2.0) * self.target_fill_rate_boost
        
        # NEW: Apply Tickr position multiplier
        regime_position_mult = self.config.regime_configs[regime].get('position_multiplier', 1.0)
        if tickr_adj['confidence'] > 65.0:
            regime_position_mult *= tickr_adj['position_mult']
        
        return {
            'bid_spread': bid_spread,
            'ask_spread': ask_spread,
            'regime': regime.value,
            'regime_target_profit': self.config.regime_configs[regime].get('target_profit_pct', 0.40),
            'regime_position_mult': regime_position_mult,
            'volume_regime': volume_regime,
            'volume_change_pct': volume_change * 100,
            'price': current_price,
            'volatility': volatility,
            'trend': trend,
            'momentum': momentum,
            'price_count': len(prices),
            'trend_deviation': trend_dev,
            'expected_profit_pct': expected_profit_pct,
            'fee_coverage': fee_coverage,
            'fee_status': fee_status,
            'inventory_status': inv_status,
            'inventory_zone': inv_zone.value,
            'inventory_imbalance': inv_imbalance,
            'hours_at_extreme': hours_at_extreme,
            'force_rebalance': force_action,
            'force_amount': force_amount,
            'force_price_adj': force_price_adj,
            'price_range_pct': price_range_pct * 100,
            'ma_cross': ma_dev,
            'current_fill_rate': current_fill_rate,
            'target_fill_rate': target_fill_rate,
            'aggression_level': self.aggression_level,
            'spread_compression_factor': self.spread_compression_factor,
            'target_fill_rate_boost': self.target_fill_rate_boost,
            'adaptive_adjustments': self.adaptive_adjustments,
            # NEW: Add Tickr-related info
            'tickr_signal': tickr_adj.get('signal_action', 'neutral'),
            'tickr_confidence': tickr_adj['confidence'],
            'tickr_reason': tickr_adj['reason']
        }
    
    def _get_halt_response(self, reason: str) -> Dict[str, Any]:
        return {
            'bid_spread': Decimal(str(self.config.absolute_max_spread)),
            'ask_spread': Decimal(str(self.config.absolute_max_spread)),
            'regime': 'HALTED',
            'regime_target_profit': 0.0,
            'regime_position_mult': 0.0,
            'halt_reason': reason,
            'price': 0.0,
            'fee_coverage': 0.0,
            'fee_status': 'HALTED',
            'inventory_status': 'HALTED',
            'inventory_zone': 'halted',
            'volume_regime': 'halted',
            'volume_change_pct': 0.0,
            'current_fill_rate': 0.0,
            'target_fill_rate': 0.0,
        }
    
    def _get_defaults(self, price_count: int = 0) -> Dict[str, Any]:
        return {
            'bid_spread': Decimal("0.00128"),
            'ask_spread': Decimal("0.00130"),
            'regime': f'warming_up_{price_count}/10',
            'regime_target_profit': 0.40,
            'regime_position_mult': 1.0,
            'volume_regime': 'unknown',
            'volume_change_pct': 0.0,
            'price': 0.0,
            'volatility': 0.0,
            'trend': 'neutral',
            'momentum': 0.0,
            'price_count': price_count,
            'trend_deviation': 0.0,
            'expected_profit_pct': 0.33,
            'fee_coverage': 1.23,
            'fee_status': 'WARMING_UP',
            'inventory_status': 'OK',
            'inventory_zone': 'green',
            'inventory_imbalance': 0.0,
            'hours_at_extreme': 0.0,
            'force_rebalance': None,
            'force_amount': 0.0,
            'force_price_adj': 0.0,
            'price_range_pct': 0.0,
            'ma_cross': 0.0,
            'current_fill_rate': 0.0,
            'target_fill_rate': 2.0,
        }

# ==================== BASE STRATEGY ====================
class BasePMM:
    def __init__(self, connectors, config, pair_name: str, trading_pair: str,
                 base_currency: str, quote_currency: str, base_order_amount: Decimal,
                 order_refresh_time: int = 300, update_interval: int = 180):
        self.connectors = connectors
        self.pair_name = pair_name
        self.trading_pair = trading_pair
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.base_order_amount = base_order_amount
        self.order_refresh_time = 300
        self.update_interval = 180
        
        # Initialize optimizer
        self.optimizer = OptimizedOptimizer(config, pair_name)
        
        # State tracking
        self.last_timestamp = 0
        self.last_update = 0
        self.last_mid_price = Decimal("0")
        
        # Statistics
        self.total_trades = 0
        self.buy_trades = 0
        self.sell_trades = 0
        self.estimated_fees = Decimal("0")
        
        # Trade cooldown tracking
        self.last_trade_time = 0
        self.min_trade_interval = 15  
        
        # MODIFIED: Add price caching to reduce API calls
        self.cached_mid_price = None
        self.last_price_update = 0
        self.price_cache_duration = 60  
        
        # MODIFIED: Add order book cache
        self.cached_order_book = None
        self.last_order_book_update = 0
        self.order_book_cache_duration = 90  

        # MODIFIED: Add adaptive refresh timing
        self.adaptive_refresh_timer = time.time()
        self.effective_refresh_time = order_refresh_time
    
    
    def on_tick(self, current_time: float):
        connector = list(self.connectors.values())[0]
        
        # Update price data
        if current_time - self.last_update >= self.update_interval:
            self._update_price_data(connector)
            self.last_update = current_time
        
        # MODIFIED: Adaptive refresh timing based on performance
        if self.optimizer.config.adaptive_refresh_rate:
            self._update_adaptive_refresh_time(current_time)
        
        # Execute strategy
        if current_time - self.last_timestamp >= self.effective_refresh_time:
            self._execute_strategy(connector)
            self.last_timestamp = current_time
    
    # MODIFIED: Add method for adaptive refresh timing
    def _update_adaptive_refresh_time(self, current_time: float):
        """Adjust refresh time based on fill rate performance"""
        if current_time - self.adaptive_refresh_timer < 300:  # Check every 5 minutes
            return
            
        self.adaptive_refresh_timer = current_time
        
        # Get current performance
        current_fill_rate = self.optimizer.get_current_fill_rate()
        target_fill_rate = 3.0  # Base target
        
        # Adjust refresh time based on performance
        if current_fill_rate < target_fill_rate * 0.5:
            # Increase trading frequency
            self.effective_refresh_time = max(self.order_refresh_time * 0.7, 30)  # Min 30 seconds
        elif current_fill_rate > target_fill_rate * 1.5:
            # Decrease trading frequency to avoid over-trading
            self.effective_refresh_time = min(self.order_refresh_time * 1.3, 300)  # Max 5 minutes
        else:
            # Normal refresh rate
            self.effective_refresh_time = self.order_refresh_time
    
    def _update_price_data(self, connector):
        try:
            current_time = time.time()
            
            # FIXED: Always try to get fresh price during warming up
            prices_count = len(self.optimizer.price_data.closes)
            if prices_count < 10:
                # Force update during warming up phase
                mid_price = connector.get_mid_price(self.trading_pair)
                if mid_price and mid_price > 0:
                    self.optimizer.add_price(float(mid_price), volume=0)
                    self.last_mid_price = mid_price
                    self.last_price_update = current_time
                    logger.info(f"{self.pair_name}: Warming up {prices_count+1}/10")
                return
            
            # Normal caching after warm up
            if current_time - self.last_price_update < self.price_cache_duration:
                return
            
            mid_price = connector.get_mid_price(self.trading_pair)
            if mid_price and mid_price > 0:
                self.optimizer.add_price(float(mid_price), volume=0)
                self.last_mid_price = mid_price
                self.last_price_update = current_time
        except Exception as e:
            logger.error(f"{self.pair_name}: Error updating price data: {e}")

    def _execute_strategy(self, connector):
        try:
            # ... existing health check code ...
            
            # Get mid price (using cache)
            current_time = time.time()
            if current_time - self.last_price_update < self.price_cache_duration:
                mid_price = self.last_mid_price
            else:
                mid_price = connector.get_mid_price(self.trading_pair)
                self.last_mid_price = mid_price
                self.last_price_update = current_time
            
            if not mid_price or mid_price <= 0:
                return
            
            inventory_pct = self._get_inventory_percentage(connector)
            market_spread = self._get_market_spread(connector)
            optimization = self.optimizer.optimize(inventory_pct, market_spread)
            
            # Handle halt
            if optimization['regime'] == 'HALTED':
                self._cancel_all_orders(connector)
                return
            
            # Execute forced rebalance if needed
            if optimization.get('force_rebalance'):
                self._execute_forced_rebalance(
                    connector,
                    optimization['force_rebalance'],
                    optimization['force_amount'],
                    optimization['force_price_adj'],
                    mid_price
                )
                return  # Don't place new orders after force rebalance
            
            # FIXED: Only cancel and replace if no active orders OR orders are stale
            active_orders = self.get_active_orders(list(self.connectors.keys())[0])
            my_orders = [o for o in active_orders if o.trading_pair == self.trading_pair]
            
            # Only refresh if we have no orders or they're too old
            if len(my_orders) == 0 or (current_time - self.last_timestamp >= self.order_refresh_time):
                self._cancel_all_orders(connector)
                self._place_orders(connector, mid_price, optimization)
                self.last_timestamp = current_time
            else:
                logger.debug(f"{self.pair_name}: Keeping existing {len(my_orders)} orders active")
                
        except Exception as e:
            logger.error(f"{self.pair_name}: Error in execute strategy: {e}")
    
    def _get_inventory_percentage(self, connector) -> float:
        try:
            # MODIFIED: Cache balance calls to reduce API usage
            base_balance = connector.get_balance(self.base_currency)
            quote_balance = connector.get_balance(self.quote_currency)
            
            # MODIFIED: Use cached price to reduce API calls
            if time.time() - self.last_price_update < self.price_cache_duration:
                mid_price = self.last_mid_price
            else:
                mid_price = connector.get_mid_price(self.trading_pair)
                self.last_mid_price = mid_price
                self.last_price_update = time.time()
            
            if not mid_price or mid_price <= 0:
                return 50.0
            
            base_value = float(base_balance) * float(mid_price)
            total_value = base_value + float(quote_balance)
            
            if total_value <= 0:
                return 50.0
            
            return (base_value / total_value) * 100.0
        except Exception:
            return 50.0
    
    # MODIFIED: Calculate effective spread based on order book depth
    def _get_market_spread(self, connector) -> float:
        try:
            # MODIFIED: Use cached order book to reduce API calls
            current_time = time.time()
            if current_time - self.last_order_book_update < self.order_book_cache_duration:
                order_book = self.cached_order_book
            else:
                order_book = connector.order_books[self.trading_pair]
                self.cached_order_book = order_book
                self.last_order_book_update = current_time
            
            if order_book and order_book.bid_entries() and order_book.ask_entries():
                # Calculate depth equivalent to 2x base_order_amount
                target_depth = float(self.base_order_amount) * 2.0
                
                # Calculate bid price at depth
                bid_cumulative = 0.0
                bid_price_at_depth = 0.0
                for entry in order_book.bid_entries():
                    bid_cumulative += float(entry.amount)
                    if bid_cumulative >= target_depth:
                        bid_price_at_depth = float(entry.price)
                        break 
                
                # Calculate ask price at depth
                ask_cumulative = 0.0
                ask_price_at_depth = 0.0
                for entry in order_book.ask_entries():
                    ask_cumulative += float(entry.amount)
                    if ask_cumulative >= target_depth:
                        ask_price_at_depth = float(entry.price)
                        break
                
                # Check if sufficient depth exists
                min_required_depth = target_depth * 0.75  # 75% of target depth
                if bid_cumulative < min_required_depth or ask_cumulative < min_required_depth:
                    logger.warning(f"{self.pair_name}: Insufficient order book depth. Bid depth: {bid_cumulative}, Ask depth: {ask_cumulative}, Required: {min_required_depth}")
                    return 0.001  # Return a high spread to signal insufficient depth
                
                # Log effective spread and depth
                if bid_price_at_depth > 0 and ask_price_at_depth > bid_price_at_depth:
                    effective_spread = (ask_price_at_depth - bid_price_at_depth) / bid_price_at_depth
                    logger.info(f"{self.pair_name}: Effective spread at depth {target_depth}: {effective_spread*100:.4f}%, Bid depth: {bid_cumulative}, Ask depth: {ask_cumulative}")
                    return effective_spread
            
            return 0.0
        except Exception as e:
            logger.error(f"{self.pair_name}: Error calculating market spread: {str(e)}")
            return 0.0
    
    def _detect_hidden_liquidity(self, order_book) -> Dict[str, Any]:
        """Detect potential hidden/iceberg orders in Kraken order book
        Returns: dict with hidden liquidity indicators"""
        try:
            if not order_book or not order_book.bid_entries() or not order_book.ask_entries():
                return {'hidden_detected': False, 'spread_adjustment': 0.0}
            
            bid_entries = list(order_book.bid_entries())
            ask_entries = list(order_book.ask_entries())
            
            # Analyze bid side for suspicious patterns
            bid_volumes = [float(entry.amount) for entry in bid_entries[:10]]
            bid_prices = [float(entry.price) for entry in bid_entries[:10]]
            
            # Analyze ask side
            ask_volumes = [float(entry.amount) for entry in ask_entries[:10]]
            ask_prices = [float(entry.price) for entry in ask_entries[:10]]
            
            hidden_detected = False
            spread_adjustment = 0.0
            
            # Detection Pattern 1: Large gaps in price levels (indicates hidden orders)
            if len(bid_prices) >= 3:
                bid_gaps = [(bid_prices[i] - bid_prices[i+1]) / bid_prices[i] 
                        for i in range(len(bid_prices)-1)]
                avg_bid_gap = statistics.mean(bid_gaps)
                max_bid_gap = max(bid_gaps)
                
                # If max gap is 3x average, likely hidden orders
                if max_bid_gap > avg_bid_gap * 3.0 and max_bid_gap > 0.001:
                    hidden_detected = True
                    spread_adjustment += 0.0002
            
            if len(ask_prices) >= 3:
                ask_gaps = [(ask_prices[i+1] - ask_prices[i]) / ask_prices[i] 
                        for i in range(len(ask_prices)-1)]
                avg_ask_gap = statistics.mean(ask_gaps)
                max_ask_gap = max(ask_gaps)
                
                if max_ask_gap > avg_ask_gap * 3.0 and max_ask_gap > 0.001:
                    hidden_detected = True
                    spread_adjustment += 0.0002
            
            # Detection Pattern 2: Unusually small orders at top of book (iceberg indicators)
            if len(bid_volumes) >= 2 and len(ask_volumes) >= 2:
                avg_bid_vol = statistics.mean(bid_volumes[1:5])  # Skip top entry
                avg_ask_vol = statistics.mean(ask_volumes[1:5])
                
                # If top orders are suspiciously small compared to deeper levels
                if bid_volumes[0] < avg_bid_vol * 0.3 and avg_bid_vol > 0:
                    hidden_detected = True
                    spread_adjustment += 0.0001
                
                if ask_volumes[0] < avg_ask_vol * 0.3 and avg_ask_vol > 0:
                    hidden_detected = True
                    spread_adjustment += 0.0001
            
            if hidden_detected:
                logger.info(f"{self.pair_name}: Hidden liquidity detected. Spread adjustment: +{spread_adjustment*100:.3f}%")
            
            return {
                'hidden_detected': hidden_detected,
                'spread_adjustment': spread_adjustment,
                'bid_depth_quality': statistics.mean(bid_volumes[:5]) if bid_volumes else 0,
                'ask_depth_quality': statistics.mean(ask_volumes[:5]) if ask_volumes else 0
            }
            
        except Exception as e:
            logger.error(f"{self.pair_name}: Error detecting hidden liquidity: {str(e)}")
            return {'hidden_detected': False, 'spread_adjustment': 0.0}

    # MODIFIED: Dynamic rebalancing with volatility-based pricing and minimum order sizes
    def _execute_forced_rebalance(self, connector, action: str, amount_pct: float, 
                                 price_adj: float, mid_price: Decimal):
        try:
            if action == 'force_sell':
                base_balance = connector.get_balance(self.base_currency)
                # Calculate 1% of available base balance
                order_size = float(base_balance) * 0.01  # MODIFIED: Increased from 0.5% to 1%
                min_size = self._get_min_order_size()
                order_size = Decimal(str(max(order_size, min_size)))
                
                # Get current volatility for dynamic price adjustment
                prices, _ = self.optimizer.price_data.get_recent(self.optimizer.config.volatility_window)
                current_volatility = self.optimizer.calculate_volatility(prices) if prices else 0.001
                
                # Dynamic price adjustment based on volatility, capped at -0.001
                volatility_factor = min(current_volatility / 0.001, 2.0)  # Cap the factor
                dynamic_price_adj = max(-0.001, -0.0008 * volatility_factor)  # MODIFIED: Increased adjustment
                
                price = mid_price * (Decimal("1") + Decimal(str(dynamic_price_adj)))
                
                self.sell(
                    connector_name=list(self.connectors.keys())[0],
                    trading_pair=self.trading_pair,
                    amount=order_size,
                    order_type=OrderType.LIMIT,
                    price=price
                )
                
            elif action == 'force_buy':
                quote_balance = connector.get_balance(self.quote_currency)
                # Calculate 1% of available quote balance
                order_value = float(quote_balance) * 0.01  # MODIFIED: Increased from 0.5% to 1%
                order_size = order_value / float(mid_price)
                min_size = self._get_min_order_size()
                order_size = Decimal(str(max(order_size, min_size)))
                
                # Get current volatility for dynamic price adjustment
                prices, _ = self.optimizer.price_data.get_recent(self.optimizer.config.volatility_window)
                current_volatility = self.optimizer.calculate_volatility(prices) if prices else 0.001
                
                # Dynamic price adjustment based on volatility, capped at +0.001
                volatility_factor = min(current_volatility / 0.001, 2.0)  # Cap the factor
                dynamic_price_adj = min(0.001, 0.0008 * volatility_factor)  # MODIFIED: Increased adjustment
                
                price = mid_price * (Decimal("1") + Decimal(str(dynamic_price_adj)))
                
                self.buy(
                    connector_name=list(self.connectors.keys())[0],
                    trading_pair=self.trading_pair,
                    amount=order_size,
                    order_type=OrderType.LIMIT,
                    price=price
                )
        except Exception as e:
            logger.error(f"{self.pair_name}: Error in forced rebalance: {str(e)}")
            pass
    
    def _get_min_order_size(self) -> float:
        """Get minimum order size based on pair"""
        min_sizes = {
            'ATOM': 7.3,
            'DOT': 1.0,
            'AVAX': 1.4,
            'ADA': 43.0,  # ADA requires larger minimum
            'LTC': 0.22,  # LTC minimum is very small
        }
        return min_sizes.get(self.base_currency, 0.001)
    
    def _place_orders(self, connector, mid_price: Decimal, optimization: Dict):
        try:
            bid_spread = optimization['bid_spread']
            ask_spread = optimization['ask_spread']
            
            # MODIFIED: Adjust spreads when order book depth is insufficient
            # MODIFIED: Use cached order book to reduce API calls
            current_time = time.time()
            if current_time - self.last_order_book_update < self.order_book_cache_duration:
                order_book = self.cached_order_book
            else:
                order_book = connector.order_books[self.trading_pair]
                self.cached_order_book = order_book
                self.last_order_book_update = current_time
                
            # Detect hidden liquidity and adjust spreads
            if order_book:
                hidden_liq = self._detect_hidden_liquidity(order_book)
                if hidden_liq['hidden_detected']:
                    adjustment = Decimal(str(hidden_liq['spread_adjustment']))
                    bid_spread += adjustment
                    ask_spread += adjustment
                    logger.info(f"{self.pair_name}: Adjusted spreads for hidden liquidity: "
                            f"{float(bid_spread)*100:.4f}%/{float(ask_spread)*100:.4f}%")

            if order_book:
                target_depth = float(self.base_order_amount) * 2.0
                bid_cumulative = sum(float(entry.amount) for entry in order_book.bid_entries())
                ask_cumulative = sum(float(entry.amount) for entry in order_book.ask_entries())
                min_required_depth = target_depth * 0.75
                
                if bid_cumulative < min_required_depth or ask_cumulative < min_required_depth:
                    bid_spread += Decimal("0.0005")  # MODIFIED: Reduced spread increase
                    ask_spread += Decimal("0.0005")  # MODIFIED: Reduced spread increase
                    logger.warning(f"{self.pair_name}: Increasing spreads due to insufficient depth. New spreads: {float(bid_spread)*100:.4f}%/{float(ask_spread)*100:.4f}%")
            
            bid_price = mid_price * (Decimal("1") - bid_spread)
            ask_price = mid_price * (Decimal("1") + ask_spread)
            
            # slippage_tolerance = Decimal("0.003")
            base_balance = connector.get_balance(self.base_currency)
            quote_balance = connector.get_balance(self.quote_currency)
            
            # Position size adjustment
            position_mult = Decimal(str(optimization.get('regime_position_mult', 1.0)))
            order_amount = self.base_order_amount * position_mult
            
            # MODIFIED: Increased position sizing for better trading activity
            #order_amount *= Decimal("1.05")
            
            # MODIFIED: Apply adaptive aggression from optimizer
            aggression_level = optimization.get('aggression_level', 1.0)
            order_amount *= Decimal(str(aggression_level))
            
            # Inventory zone adjustment
            inv_zone = optimization.get('inventory_zone', 'green')
            if inv_zone == 'critical':
                order_amount *= Decimal("2.0")  # MODIFIED: Increased from 1.5 to 2.0
            elif inv_zone == 'red':
                order_amount *= Decimal("1.5")  # MODIFIED: Increased from 1.2 to 1.5
            
            # Trend-based order placement
            regime = optimization.get('regime', 'normal')
            inventory_imb = optimization.get('inventory_imbalance', 0.0)
            
            place_buy = True
            place_sell = True
            
            # MODIFIED: More aggressive entry during trending conditions
            if regime == 'trending_down' and inventory_imb > 2.0:
                place_buy = False
                order_amount *= Decimal("2.0")  # MODIFIED: Increased from 1.5 to 2.0
            elif regime == 'trending_up' and inventory_imb < -2.0:
                place_sell = False
                order_amount *= Decimal("2.0")  # MODIFIED: Increased from 1.5 to 2.0
            
            # MODIFIED: Check if profit is sufficient to cover fees (relaxed requirement)
            avg_spread = (bid_spread + ask_spread) / 2
            kraken_fee_rate = Decimal("0.0026")  # 0.26% Kraken fee
            expected_profit = avg_spread * 2
            if expected_profit <= kraken_fee_rate * Decimal("1.15"):  # Reduced from 1.3 to 1.15
                return  # Skip placing orders if profit is not sufficient
            
            # MODIFIED: Add cooldown check to prevent over-trading (reduced interval)
            current_time = time.time()
            if current_time - self.last_trade_time < self.min_trade_interval:
                return  # Skip placing orders if within cooldown period
            
            # Place buy order
            if place_buy:
                buy_amount_needed = order_amount * bid_price
                if quote_balance >= buy_amount_needed:
                    self.buy(
                        connector_name=list(self.connectors.keys())[0],
                        trading_pair=self.trading_pair,
                        amount=order_amount,
                        order_type=OrderType.LIMIT,
                        price=bid_price
                    )
            
            # Place sell order
            if place_sell:
                if base_balance >= order_amount:
                    self.sell(
                        connector_name=list(self.connectors.keys())[0],
                        trading_pair=self.trading_pair,
                        amount=order_amount,
                        order_type=OrderType.LIMIT,
                        price=ask_price
                    )
        except Exception as e:
            logger.error(f"{self.pair_name}: Error placing orders: {str(e)}")
            pass
    
    def buy(self, connector_name: str, trading_pair: str, amount: Decimal, order_type: OrderType, price: Decimal):
        """Place a buy order"""
        try:
            connector = self.connectors[connector_name]
            connector.buy(trading_pair=trading_pair, amount=amount, order_type=OrderType.LIMIT, price=price)
            # MODIFIED: Add logging for trade execution
            logger.info(f"{self.pair_name} BUY order placed: {amount} at {price}")
        except Exception:
            pass
    
    def sell(self, connector_name: str, trading_pair: str, amount: Decimal, order_type: OrderType, price: Decimal):
        """Place a sell order"""
        try:
            connector = self.connectors[connector_name]
            connector.sell(trading_pair=trading_pair, amount=amount, order_type=OrderType.LIMIT, price=price)
            # MODIFIED: Add logging for trade execution
            logger.info(f"{self.pair_name} SELL order placed: {amount} at {price}")
        except Exception:
            pass
    
    def did_fill_order(self, event):
        try:
            if event.trading_pair != self.trading_pair:
                return
            
            self.total_trades += 1
            self.optimizer.track_fill()
            
            if event.trade_type.name == "BUY":
                self.buy_trades += 1
                # MODIFIED: Add detailed logging for filled buy orders
                logger.info(f"{self.pair_name} BUY order filled: {event.amount} at {event.price} - Timestamp: {time.time()}")
            else:
                self.sell_trades += 1
                # MODIFIED: Add detailed logging for filled sell orders
                logger.info(f"{self.pair_name} SELL order filled: {event.amount} at {event.price} - Timestamp: {time.time()}")
            
            # Calculate fees
            trade_value = float(event.amount) * float(event.price)
            fee = Decimal(str(trade_value * 0.0026))
            self.estimated_fees += fee
            
            # MODIFIED: Update last trade time for cooldown tracking
            self.last_trade_time = time.time()
            
        except Exception:
            pass
    
    def _cancel_all_orders(self, connector):
        try:
            connector_name = list(self.connectors.keys())[0]
            for order in self.get_active_orders(connector_name):
                if order.trading_pair == self.trading_pair:
                    self.cancel(
                        connector_name=connector_name,
                        trading_pair=order.trading_pair,
                        order_id=order.client_order_id
                    )
        except Exception:
            pass
    
    def get_active_orders(self, connector_name: str):
        """Get active orders from connector"""
        try:
            connector = self.connectors[connector_name]
            return list(connector.in_flight_orders.values())
        except Exception:
            return []
    
    def cancel(self, connector_name: str, trading_pair: str, order_id: str):
        """Cancel an order"""
        try:
            connector = self.connectors[connector_name]
            connector.cancel(trading_pair, order_id)
        except Exception:
            pass
    
    def format_status(self) -> str:
        try:
            connector = list(self.connectors.values())[0]
            
            # Use cached price to reduce API calls
            if time.time() - self.last_price_update < self.price_cache_duration:
                mid_price = self.last_mid_price
            else:
                mid_price = connector.get_mid_price(self.trading_pair)
                self.last_mid_price = mid_price
                self.last_price_update = time.time()
            
            if not mid_price or mid_price <= 0:
                return f"Waiting for {self.pair_name} market data..."
            
            inventory_pct = self._get_inventory_percentage(connector)
            market_spread = self._get_market_spread(connector)
            optimization = self.optimizer.optimize(inventory_pct, market_spread)
            
            # Format price based on pair
            if self.pair_name in ['ATOM', 'AVAX', 'DOT']:
                price_str = f"${mid_price:.3f}"
            elif self.pair_name in ['ADA']:  
                price_str = f"${mid_price:.4f}"
            elif self.pair_name in ['LTC']:
                price_str = f"${mid_price:.2f}"
            else:
                price_str = f"${mid_price:.2f}"
            
            # Inventory emoji
            inv_zone = optimization['inventory_zone']
            inv_emoji = {
                'green': '✅',
                'yellow': '⚠️',
                'red': '🔴',
                'critical': '🚨'
            }.get(inv_zone, '❓')
            
            # Fee emoji
            fee_cov = optimization['fee_coverage']
            fee_emoji = '🔴' if fee_cov < 1.15 else '⚠️' if fee_cov < 1.3 else '✅'
            
            # Build status
            status_parts = [
                f"{self.pair_name}: {price_str}",
                f"Regime: {optimization['regime'].upper()}",
                f"Inv: {inventory_pct:.1f}% {inv_emoji}",
                f"Spreads: {float(optimization['bid_spread'])*100:.2f}%/{float(optimization['ask_spread'])*100:.2f}%",
                f"Fees: {fee_cov:.2f}x {fee_emoji}",
            ]
            
            # Fill rate
            fill_rate = optimization['current_fill_rate']
            target_fill = optimization['target_fill_rate']
            if fill_rate > target_fill * 1.2:
                status_parts.append(f"Fills: {fill_rate:.1f}/hr ⚠️")
            else:
                status_parts.append(f"Fills: {fill_rate:.1f}/hr")
            
            # Trades
            status_parts.append(f"Trades: {self.buy_trades}/{self.sell_trades}")
            
            # Adaptive parameters
            aggression = optimization.get('aggression_level', 1.0)
            if aggression > 1.2:
                status_parts.append(f"Aggr: {aggression:.1f}x 🚀")
            elif aggression < 0.8:
                status_parts.append(f"Aggr: {aggression:.1f}x 🐢")
            
            # NEW: Tickr signal info
            if TICKR_AVAILABLE:
                tickr_signal = optimization.get('tickr_signal', 'neutral')
                tickr_conf = optimization.get('tickr_confidence', 0)
                
                if tickr_conf > 70:
                    signal_emoji = '📈' if 'buy' in tickr_signal else '📉' if 'sell' in tickr_signal else '➡️'
                    status_parts.append(f"Signal: {tickr_signal.upper()} {signal_emoji}")
                elif tickr_conf > 50:
                    status_parts.append(f"Signal: {tickr_signal}")
            
            # Special conditions
            if optimization.get('force_rebalance'):
                status_parts.append("🚨 REBALANCING")
            if optimization['regime'] == 'HALTED':
                status_parts.append("⛔ HALTED")
            
            return " | ".join(status_parts)
            
        except Exception as e:
            return f"Error getting {self.pair_name} status: {str(e)}"

# ==================== MAIN STRATEGY ====================
class Multi_pmm(ScriptStrategyBase):
    markets = {
        "kraken_paper_trade": {"ATOM-USDT", "AVAX-USDT", "DOT-USDT", "ADA-USDT", "LTC-USDT"}  
    }
    
    def __init__(self, connectors):
        super().__init__(connectors)
        
        if TICKR_AVAILABLE:
            logger.info("=== TICKR INTEGRATION CHECK ===")
            try:
                test_client = TickrClient()
                # Test with ATOM as sample
                df = test_client.get_candles('ATOM/USDT', '15m', limit=5)
                if df is not None and len(df) > 0:
                    logger.info(f"✅ Tickr working: Retrieved {len(df)} ATOM candles")
                    logger.info(f"Latest candle: {df.tail(1)[['open', 'high', 'low', 'close', 'volume']].to_string()}")
                else:
                    logger.warning("⚠️ Tickr test returned no data")
            except Exception as e:
                logger.error(f"❌ Tickr test failed: {e}")
        else:
            logger.warning("⚠️ Tickr not available")

        # MODIFIED: Define strategies first before accessing them
        # ATOM Configuration - Optimized for moderate volatility with balanced spreads
        # MODIFIED: Added ATOM configuration based on market analysis
        # MODIFIED: Increased spreads by 20% for ATOM (higher volatility) and set inventory_adjustment_aggression to 1.2
        ATOM_CONFIG = BaseConfig(
            absolute_min_spread=0.0026,  # Kraken fee
            absolute_max_spread=0.015,
            volatility_extreme_threshold=0.003,
            volatility_high_threshold=0.0015,
            volatility_low_threshold=0.0008,
            volatility_ultra_low_threshold=0.0004,
            trend_strong_threshold=0.0010,
            momentum_strong_threshold=0.0008,
            base_inv_adjustment=0.0005,
            max_inv_adjustment=0.0025,
            hours_at_extreme_threshold=2.0,
            emergency_halt_threshold=0.040,  
            volume_surge_threshold=0.25,
            volume_collapse_threshold=-0.25,
            price_drop_large=0.012,
            price_rise_large=0.012,
            volatility_add_per_0001=0.00030,
            market_spread_add_factor=0.60,
            inventory_adjustment_aggression=1.0,  
            aggressive_spread_compression=False,  
            adaptive_refresh_rate=True,  
            regime_configs={
                MarketRegime.ULTRA_CALM: {
                    'base_bid_spread': 0.0027,  # Reduced from 0.0028 (-3.6%)
                    'base_ask_spread': 0.0027,  # Reduced from 0.0028
                    'min_spread': 0.0026,
                    'position_multiplier': 1.2,
                    'target_fill_rate': 2.5,
                    'inventory_max_imbalance': 8.0,
                    'target_profit_pct': 0.58,
                },
                MarketRegime.CALM: {
                    'base_bid_spread': 0.0028,  # Reduced from 0.0030 (-6.7%)
                    'base_ask_spread': 0.0028,  # Reduced from 0.0030
                    'min_spread': 0.0026,
                    'position_multiplier': 1.1,
                    'target_fill_rate': 2.8,
                    'inventory_max_imbalance': 9.0,
                    'target_profit_pct': 0.60,
                },
                MarketRegime.NORMAL: {
                    'base_bid_spread': 0.0030,  # Reduced from 0.0033 (-9%)
                    'base_ask_spread': 0.0030,  # Reduced from 0.0033
                    'min_spread': 0.0026,
                    'position_multiplier': 1.0,
                    'target_fill_rate': 3.0,
                    'inventory_max_imbalance': 9.0,
                    'target_profit_pct': 0.64,
                },
                MarketRegime.VOLATILE: {
                    'base_bid_spread': 0.0036,  # Reduced from 0.0040 (-10%)
                    'base_ask_spread': 0.0036,  # Reduced from 0.0040
                    'min_spread': 0.0026,
                    'position_multiplier': 0.8,
                    'target_fill_rate': 1.5,
                    'inventory_max_imbalance': 6.0,
                    'target_profit_pct': 0.80,
                },
                MarketRegime.EXTREME: {
                    'base_bid_spread': 0.0048,  # Reduced from 0.0055 (-13%)
                    'base_ask_spread': 0.0048,  # Reduced from 0.0055
                    'min_spread': 0.0026,
                    'position_multiplier': 0.5,
                    'target_fill_rate': 0.8,
                    'inventory_max_imbalance': 4.0,
                    'target_profit_pct': 1.10,
                },
                MarketRegime.TRENDING_UP: {
                    'base_bid_spread': 0.0032,  # Reduced from 0.0036 (-11%)
                    'base_ask_spread': 0.0042,  # Reduced from 0.0048 (-12%)
                    'min_spread': 0.0026,
                    'position_multiplier': 0.7,
                    'target_fill_rate': 1.0,
                    'inventory_max_imbalance': 3.0,
                    'target_profit_pct': 1.05,
                    'inventory_hard_floor': 46.0,
                },
                MarketRegime.TRENDING_DOWN: {
                    'base_bid_spread': 0.0042,  # Reduced from 0.0048 (-12%)
                    'base_ask_spread': 0.0032,  # Reduced from 0.0036 (-11%)
                    'min_spread': 0.0026,
                    'position_multiplier': 0.7,
                    'target_fill_rate': 1.0,
                    'inventory_max_imbalance': 3.0,
                    'target_profit_pct': 0.90,
                    'inventory_hard_ceiling': 54.0,
                },
                MarketRegime.CHOPPY: {
                    'base_bid_spread': 0.0037,  # Reduced from 0.0042 (-12%)
                    'base_ask_spread': 0.0037,  # Reduced from 0.0042
                    'min_spread': 0.0026,
                    'position_multiplier': 0.9,
                    'target_fill_rate': 1.8,
                    'inventory_max_imbalance': 7.0,
                    'target_profit_pct': 0.74,
                }
            }
        )
        
        ADA_CONFIG = BaseConfig(
            absolute_min_spread=0.0026,
            absolute_max_spread=0.020,  # Lower max for stable coin
            volatility_extreme_threshold=0.004,  # Less volatile than others
            volatility_high_threshold=0.0018,
            volatility_low_threshold=0.0007,
            volatility_ultra_low_threshold=0.0003,
            trend_strong_threshold=0.0008,
            momentum_strong_threshold=0.0006,
            base_inv_adjustment=0.0004,
            max_inv_adjustment=0.0020,
            hours_at_extreme_threshold=2.0,
            emergency_halt_threshold=0.050,
            volume_surge_threshold=0.25,
            volume_collapse_threshold=-0.25,
            price_drop_large=0.010,
            price_rise_large=0.010,
            volatility_add_per_0001=0.00030,
            market_spread_add_factor=0.60,
            inventory_adjustment_aggression=0.85,  # Slightly less aggressive
            aggressive_spread_compression=False,
            adaptive_refresh_rate=True,
            regime_configs={
                MarketRegime.ULTRA_CALM: {
                    'base_bid_spread': 0.0027,  # Reduced from 0.0028
                    'base_ask_spread': 0.0027,  # Reduced from 0.0030 (also made symmetric)
                    'min_spread': 0.0026,
                    'position_multiplier': 1.3,
                    'target_fill_rate': 3.0,
                    'inventory_max_imbalance': 9.0,
                    'target_profit_pct': 0.60,
                },
                MarketRegime.CALM: {
                    'base_bid_spread': 0.0028,  # Reduced from 0.0031
                    'base_ask_spread': 0.0028,  # Reduced from 0.0033 (also made symmetric)
                    'min_spread': 0.0026,
                    'position_multiplier': 1.2,
                    'target_fill_rate': 3.3,
                    'inventory_max_imbalance': 10.0,
                    'target_profit_pct': 0.65,
                },
                MarketRegime.NORMAL: {
                    'base_bid_spread': 0.0030,  # Reduced from 0.0034
                    'base_ask_spread': 0.0030,  # Reduced from 0.0036 (also made symmetric)
                    'min_spread': 0.0026,
                    'position_multiplier': 1.1,
                    'target_fill_rate': 3.5,
                    'inventory_max_imbalance': 10.0,
                    'target_profit_pct': 0.70,
                },
                MarketRegime.VOLATILE: {
                    'base_bid_spread': 0.0036,  # Reduced from 0.0040
                    'base_ask_spread': 0.0036,  # Reduced from 0.0042 (also made symmetric)
                    'min_spread': 0.0026,
                    'position_multiplier': 0.85,
                    'target_fill_rate': 1.8,
                    'inventory_max_imbalance': 7.0,
                    'target_profit_pct': 0.85,
                },
                MarketRegime.EXTREME: {
                    'base_bid_spread': 0.0048,  # Reduced from 0.0055
                    'base_ask_spread': 0.0048,  # Reduced from 0.0055
                    'min_spread': 0.0026,
                    'position_multiplier': 0.6,
                    'target_fill_rate': 1.0,
                    'inventory_max_imbalance': 5.0,
                    'target_profit_pct': 1.15,
                },
                MarketRegime.TRENDING_UP: {
                    'base_bid_spread': 0.0032,  # Reduced from 0.0036
                    'base_ask_spread': 0.0044,  # Reduced from 0.0050
                    'min_spread': 0.0026,
                    'position_multiplier': 0.75,
                    'target_fill_rate': 1.2,
                    'inventory_max_imbalance': 4.0,
                    'target_profit_pct': 0.95,
                    'inventory_hard_floor': 46.0,
                },
                MarketRegime.TRENDING_DOWN: {
                    'base_bid_spread': 0.0044,  # Reduced from 0.0050
                    'base_ask_spread': 0.0032,  # Reduced from 0.0036
                    'min_spread': 0.0026,
                    'position_multiplier': 0.75,
                    'target_fill_rate': 1.2,
                    'inventory_max_imbalance': 4.0,
                    'target_profit_pct': 0.95,
                    'inventory_hard_ceiling': 54.0,
                },
                MarketRegime.CHOPPY: {
                    'base_bid_spread': 0.0039,  # Reduced from 0.0045
                    'base_ask_spread': 0.0039,  # Reduced from 0.0047 (also made symmetric)
                    'min_spread': 0.0026,
                    'position_multiplier': 0.95,
                    'target_fill_rate': 2.0,
                    'inventory_max_imbalance': 8.0,
                    'target_profit_pct': 0.80,
                }
            }
        )
        

        # AVAX Configuration - Optimized for high volatility with wider spreads
        AVAX_CONFIG = BaseConfig(
            absolute_min_spread=0.0026,  # Kraken fee
            absolute_max_spread=0.015,
            volatility_extreme_threshold=0.003,
            volatility_high_threshold=0.0015,
            volatility_low_threshold=0.0008,
            volatility_ultra_low_threshold=0.0004,
            trend_strong_threshold=0.0010,
            momentum_strong_threshold=0.0009,
            base_inv_adjustment=0.0005,
            max_inv_adjustment=0.0025,
            hours_at_extreme_threshold=2.0,
            emergency_halt_threshold=0.040,
            volume_surge_threshold=0.25,
            volume_collapse_threshold=-0.25,
            price_drop_large=0.012,
            price_rise_large=0.012,
            volatility_add_per_0001=0.0004,
            market_spread_add_factor=0.65,
            inventory_adjustment_aggression=1.0,
            aggressive_spread_compression=False,
            adaptive_refresh_rate=True,
            regime_configs={
                MarketRegime.ULTRA_CALM: {
                    'base_bid_spread': 0.0029,  # Reduced from 0.0032
                    'base_ask_spread': 0.0029,  # Reduced from 0.0034 (also made symmetric)
                    'min_spread': 0.0026,
                    'position_multiplier': 1.2,
                    'target_fill_rate': 2.5,
                    'inventory_max_imbalance': 8.0,
                    'target_profit_pct': 0.68,
                },
                MarketRegime.CALM: {
                    'base_bid_spread': 0.0031,  # Reduced from 0.0035
                    'base_ask_spread': 0.0031,  # Reduced from 0.0037 (also made symmetric)
                    'min_spread': 0.0026,
                    'position_multiplier': 1.1,
                    'target_fill_rate': 2.8,
                    'inventory_max_imbalance': 9.0,
                    'target_profit_pct': 0.72,
                },
                MarketRegime.NORMAL: {
                    'base_bid_spread': 0.0034,  # Reduced from 0.0038
                    'base_ask_spread': 0.0034,  # Reduced from 0.0040 (also made symmetric)
                    'min_spread': 0.0026,
                    'position_multiplier': 1.0,
                    'target_fill_rate': 3.0,
                    'inventory_max_imbalance': 9.0,
                    'target_profit_pct': 0.76,
                },
                MarketRegime.VOLATILE: {
                    'base_bid_spread': 0.0040,  # Reduced from 0.0045
                    'base_ask_spread': 0.0040,  # Reduced from 0.0047 (also made symmetric)
                    'min_spread': 0.0026,
                    'position_multiplier': 0.8,
                    'target_fill_rate': 1.5,
                    'inventory_max_imbalance': 6.0,
                    'target_profit_pct': 0.95,
                },
                MarketRegime.EXTREME: {
                    'base_bid_spread': 0.0052,  # Reduced from 0.0060
                    'base_ask_spread': 0.0052,  # Reduced from 0.0060
                    'min_spread': 0.0026,
                    'position_multiplier': 0.5,
                    'target_fill_rate': 0.8,
                    'inventory_max_imbalance': 4.0,
                    'target_profit_pct': 1.25,
                },
                MarketRegime.TRENDING_UP: {
                    'base_bid_spread': 0.0035,  # Reduced from 0.0040
                    'base_ask_spread': 0.0048,  # Reduced from 0.0055
                    'min_spread': 0.0026,
                    'position_multiplier': 0.7,
                    'target_fill_rate': 1.0,
                    'inventory_max_imbalance': 3.0,
                    'target_profit_pct': 1.05,
                    'inventory_hard_floor': 46.0,
                },
                MarketRegime.TRENDING_DOWN: {
                    'base_bid_spread': 0.0048,  # Reduced from 0.0055
                    'base_ask_spread': 0.0035,  # Reduced from 0.0040
                    'min_spread': 0.0026,
                    'position_multiplier': 0.7,
                    'target_fill_rate': 1.0,
                    'inventory_max_imbalance': 3.0,
                    'target_profit_pct': 1.05,
                    'inventory_hard_ceiling': 54.0,
                },
                MarketRegime.CHOPPY: {
                    'base_bid_spread': 0.0044,  # Reduced from 0.0050
                    'base_ask_spread': 0.0044,  # Reduced from 0.0052 (also made symmetric)
                    'min_spread': 0.0026,
                    'position_multiplier': 0.9,
                    'target_fill_rate': 1.8,
                    'inventory_max_imbalance': 7.0,
                    'target_profit_pct': 0.88,
                }
            }
        )
        # DOT Configuration - Optimized for moderate-high volatility
        DOT_CONFIG = BaseConfig(
            absolute_min_spread=0.0035,
            absolute_max_spread=0.0150,
            volatility_extreme_threshold=0.0028,
            volatility_high_threshold=0.0014,
            volatility_low_threshold=0.0007,
            volatility_ultra_low_threshold=0.0004,
            trend_strong_threshold=0.0010,
            momentum_strong_threshold=0.0008,
            base_inv_adjustment=0.0005,
            max_inv_adjustment=0.0025,
            hours_at_extreme_threshold=2.0,
            emergency_halt_threshold=0.040,
            volume_surge_threshold=0.25,
            volume_collapse_threshold=-0.25,
            price_drop_large=0.012,
            price_rise_large=0.012,
            volatility_add_per_0001=0.00040,
            market_spread_add_factor=0.65,
            inventory_adjustment_aggression=0.9,
            aggressive_spread_compression=False,
            adaptive_refresh_rate=True,
            regime_configs={
                MarketRegime.ULTRA_CALM: {
                    'base_bid_spread': 0.0032,  # Reduced from 0.0035
                    'base_ask_spread': 0.0032,  # Reduced from 0.0037 (also made symmetric)
                    'min_spread': 0.0030,  # Reduced from 0.0035
                    'position_multiplier': 1.1,
                    'target_fill_rate': 2.3,
                    'inventory_max_imbalance': 8.0,
                    'target_profit_pct': 0.70,
                },
                MarketRegime.CALM: {
                    'base_bid_spread': 0.0034,  # Reduced from 0.0038
                    'base_ask_spread': 0.0034,  # Reduced from 0.0040 (also made symmetric)
                    'min_spread': 0.0030,  # Reduced from 0.0035
                    'position_multiplier': 1.0,
                    'target_fill_rate': 2.6,
                    'inventory_max_imbalance': 9.0,
                    'target_profit_pct': 0.75,
                },
                MarketRegime.NORMAL: {
                    'base_bid_spread': 0.0037,  # Reduced from 0.0042
                    'base_ask_spread': 0.0037,  # Reduced from 0.0044 (also made symmetric)
                    'min_spread': 0.0030,  # Reduced from 0.0035
                    'position_multiplier': 1.0,
                    'target_fill_rate': 2.8,
                    'inventory_max_imbalance': 9.0,
                    'target_profit_pct': 0.80,
                },
                MarketRegime.VOLATILE: {
                    'base_bid_spread': 0.0044,  # Reduced from 0.0050
                    'base_ask_spread': 0.0044,  # Reduced from 0.0052 (also made symmetric)
                    'min_spread': 0.0030,  # Reduced from 0.0035
                    'position_multiplier': 0.75,
                    'target_fill_rate': 1.4,
                    'inventory_max_imbalance': 6.0,
                    'target_profit_pct': 1.00,
                },
                MarketRegime.EXTREME: {
                    'base_bid_spread': 0.0056,  # Reduced from 0.0065
                    'base_ask_spread': 0.0056,  # Reduced from 0.0065
                    'min_spread': 0.0030,  # Reduced from 0.0035
                    'position_multiplier': 0.5,
                    'target_fill_rate': 0.8,
                    'inventory_max_imbalance': 4.0,
                    'target_profit_pct': 1.30,
                },
                MarketRegime.TRENDING_UP: {
                    'base_bid_spread': 0.0038,  # Reduced from 0.0044
                    'base_ask_spread': 0.0050,  # Reduced from 0.0058
                    'min_spread': 0.0030,  # Reduced from 0.0035
                    'position_multiplier': 0.7,
                    'target_fill_rate': 1.0,
                    'inventory_max_imbalance': 3.0,
                    'target_profit_pct': 1.10,
                    'inventory_hard_floor': 46.0,
                },
                MarketRegime.TRENDING_DOWN: {
                    'base_bid_spread': 0.0050,  # Reduced from 0.0058
                    'base_ask_spread': 0.0038,  # Reduced from 0.0044
                    'min_spread': 0.0030,  # Reduced from 0.0035
                    'position_multiplier': 0.7,
                    'target_fill_rate': 1.0,
                    'inventory_max_imbalance': 3.0,
                    'target_profit_pct': 1.10,
                    'inventory_hard_ceiling': 54.0,
                },
                MarketRegime.CHOPPY: {
                    'base_bid_spread': 0.0047,  # Reduced from 0.0054
                    'base_ask_spread': 0.0047,  # Reduced from 0.0056 (also made symmetric)
                    'min_spread': 0.0030,  # Reduced from 0.0035
                    'position_multiplier': 0.85,
                    'target_fill_rate': 1.7,
                    'inventory_max_imbalance': 7.0,
                    'target_profit_pct': 0.92,
                }
            }
        )

        # LTC is moderately volatile with good liquidity
        LTC_CONFIG = BaseConfig(
            absolute_min_spread=0.0026,
            absolute_max_spread=0.018,
            volatility_extreme_threshold=0.0035,
            volatility_high_threshold=0.0016,
            volatility_low_threshold=0.0008,
            volatility_ultra_low_threshold=0.0004,
            trend_strong_threshold=0.0009,
            momentum_strong_threshold=0.0007,
            base_inv_adjustment=0.00045,
            max_inv_adjustment=0.0022,
            hours_at_extreme_threshold=2.0,
            emergency_halt_threshold=0.045,
            volume_surge_threshold=0.25,
            volume_collapse_threshold=-0.25,
            price_drop_large=0.011,
            price_rise_large=0.011,
            volatility_add_per_0001=0.00033,
            market_spread_add_factor=0.62,
            inventory_adjustment_aggression=0.9,
            aggressive_spread_compression=False,
            adaptive_refresh_rate=True,
            regime_configs={
                MarketRegime.ULTRA_CALM: {
                    'base_bid_spread': 0.0027,  # Reduced from 0.0030
                    'base_ask_spread': 0.0027,  # Reduced from 0.0032 (also made symmetric)
                    'min_spread': 0.0026,
                    'position_multiplier': 1.2,
                    'target_fill_rate': 2.8,
                    'inventory_max_imbalance': 8.5,
                    'target_profit_pct': 0.65,
                },
                MarketRegime.CALM: {
                    'base_bid_spread': 0.0029,  # Reduced from 0.0033
                    'base_ask_spread': 0.0029,  # Reduced from 0.0035 (also made symmetric)
                    'min_spread': 0.0026,
                    'position_multiplier': 1.15,
                    'target_fill_rate': 3.0,
                    'inventory_max_imbalance': 9.5,
                    'target_profit_pct': 0.70,
                },
                MarketRegime.NORMAL: {
                    'base_bid_spread': 0.0032,  # Reduced from 0.0036
                    'base_ask_spread': 0.0032,  # Reduced from 0.0038 (also made symmetric)
                    'min_spread': 0.0026,
                    'position_multiplier': 1.05,
                    'target_fill_rate': 3.2,
                    'inventory_max_imbalance': 9.5,
                    'target_profit_pct': 0.75,
                },
                MarketRegime.VOLATILE: {
                    'base_bid_spread': 0.0038,  # Reduced from 0.0043
                    'base_ask_spread': 0.0038,  # Reduced from 0.0045 (also made symmetric)
                    'min_spread': 0.0026,
                    'position_multiplier': 0.8,
                    'target_fill_rate': 1.6,
                    'inventory_max_imbalance': 6.5,
                    'target_profit_pct': 0.90,
                },
                MarketRegime.EXTREME: {
                    'base_bid_spread': 0.0050,  # Reduced from 0.0058
                    'base_ask_spread': 0.0050,  # Reduced from 0.0058
                    'min_spread': 0.0026,
                    'position_multiplier': 0.55,
                    'target_fill_rate': 0.9,
                    'inventory_max_imbalance': 4.5,
                    'target_profit_pct': 1.20,
                },
                MarketRegime.TRENDING_UP: {
                    'base_bid_spread': 0.0038,
                    'base_ask_spread': 0.0052,
                    'min_spread': 0.0026,
                    'position_multiplier': 0.7,
                    'target_fill_rate': 1.1,
                    'inventory_max_imbalance': 3.5,
                    'target_profit_pct': 1.00,
                    'inventory_hard_floor': 46.0,
                },
                MarketRegime.TRENDING_DOWN: {
                    'base_bid_spread': 0.0052,
                    'base_ask_spread': 0.0038,
                    'min_spread': 0.0026,
                    'position_multiplier': 0.7,
                    'target_fill_rate': 1.1,
                    'inventory_max_imbalance': 3.5,
                    'target_profit_pct': 1.00,
                    'inventory_hard_ceiling': 54.0,
                },
                MarketRegime.CHOPPY: {
                    'base_bid_spread': 0.0048,
                    'base_ask_spread': 0.0050,
                    'min_spread': 0.0026,
                    'position_multiplier': 0.9,
                    'target_fill_rate': 1.9,
                    'inventory_max_imbalance': 7.5,
                    'target_profit_pct': 0.85,
                }
            }
        )
        
        
        self.strategies = {
            "ATOM": BasePMM(  
                connectors=connectors,
                config=ATOM_CONFIG,
                pair_name="ATOM",
                trading_pair="ATOM-USDT",
                base_currency="ATOM",
                quote_currency="USDT",
                base_order_amount=Decimal("73.2"),  
                order_refresh_time=300,  # MODIFIED: Increased to 180 seconds to reduce API calls
                update_interval=180  # MODIFIED: Increased to 120 seconds
            ),
            "ADA": BasePMM(  
                connectors=connectors,
                config=ADA_CONFIG,
                pair_name="ADA",
                trading_pair="ADA-USDT",
                base_currency="ADA",
                quote_currency="USDT",
                base_order_amount=Decimal("430.0"),  # Adjusted for risk management
                order_refresh_time=300,  # MODIFIED: Increased to 180 seconds to reduce API calls
                update_interval=180  # MODIFIED: Increased to 150 seconds
            ),
            "AVAX": BasePMM(
                connectors=connectors,
                config=AVAX_CONFIG,
                pair_name="AVAX",
                trading_pair="AVAX-USDT",
                base_currency="AVAX",
                quote_currency="USDT",
                base_order_amount=Decimal("14.0"),  # Reduced for risk management
                order_refresh_time=300,  # MODIFIED: Increased to 180 seconds to reduce API calls
                update_interval=180  # MODIFIED: Increased to 150 seconds
            ),
            "LTC": BasePMM(  
                connectors=connectors,
                config=LTC_CONFIG,
                pair_name="LTC",
                trading_pair="LTC-USDT",
                base_currency="LTC",
                quote_currency="USDT",
                base_order_amount=Decimal("2.16"),  # Reduced for risk management
                order_refresh_time=300,  # MODIFIED: Increased to 180 seconds to reduce API calls
                update_interval=180  # MODIFIED: Increased to 150 seconds
            ),
            "DOT": BasePMM(
                connectors=connectors,
                config=DOT_CONFIG,
                pair_name="DOT",
                trading_pair="DOT-USDT",
                base_currency="DOT",
                quote_currency="USDT",
                base_order_amount=Decimal("74.0"),
                order_refresh_time=300,
                update_interval=180
            ),
            
            
        }
        
        # MODIFIED: Initialize last_execution_times AFTER strategies are defined
        self.last_execution_times = {pair: 0 for pair in self.strategies}
        
        # MODIFIED: Initialize risk management variables AFTER strategies are defined
        self.initial_balance = self._calculate_initial_balance(connectors)
        self.daily_pnl = Decimal("0")
        self.max_daily_loss = self.initial_balance * Decimal("0.08")  # MODIFIED: Increased from 5% to 8%
        self.halt_until = 0
        self.last_reset_time = time.time()
    
    # MODIFIED: Calculate initial account balance for risk management
    def _calculate_initial_balance(self, connectors):
        total_balance = Decimal("0")
        connector = list(connectors.values())[0]
        
        try:
            # Get quote balance once
            quote_balance = Decimal(str(connector.get_balance('USDT')))
            total_balance += quote_balance
            
            # Add value of each base currency
            for pair_name, strategy in self.strategies.items():
                try:
                    base_balance = connector.get_balance(strategy.base_currency)
                    # MODIFIED: Use cached price to reduce API calls
                    if time.time() - strategy.last_price_update < strategy.price_cache_duration:
                        mid_price = strategy.last_mid_price
                    else:
                        mid_price = connector.get_mid_price(strategy.trading_pair)
                        strategy.last_mid_price = mid_price
                        strategy.last_price_update = time.time()
                    
                    if mid_price and mid_price > 0:
                        base_value = Decimal(str(base_balance)) * mid_price
                        total_balance += base_value
                        logger.info(f"{pair_name}: {base_balance} @ ${float(mid_price):.4f} = ${float(base_value):.2f}")
                except Exception as e:
                    logger.error(f"Error calculating balance for {pair_name}: {str(e)}")
                    pass
            
            logger.info(f"Total portfolio value: ${float(total_balance):.2f}")
            
        except Exception as e:
            logger.error(f"Error calculating initial balance: {str(e)}")
            pass
        
        return total_balance if total_balance > 0 else Decimal("1000")  # Default fallback
    
    # MODIFIED: Add method to reset daily PnL at midnight UTC
    def _reset_daily_pnl_if_needed(self):
        current_time = time.time()
        # Check if 24 hours have passed since last reset
        if current_time - self.last_reset_time >= 86400:  # 24 hours in seconds
            self.daily_pnl = Decimal("0")
            self.last_reset_time = current_time
            logger.info("Daily PnL reset at midnight UTC")
    
    def on_tick(self):
        """Execute all strategies but don't run them all at exact same time"""
        current_time = time.time()
        
        # Check if trading is halted
        if self.halt_until > current_time:
            if int(current_time) % 60 == 0:
                logger.warning(f"Trading halted until {datetime.datetime.fromtimestamp(self.halt_until).strftime('%Y-%m-%d %H:%M:%S')}")
            return
        
        # Reset daily PnL if needed
        self._reset_daily_pnl_if_needed()
        
        # FIXED: Execute all strategies, but spread them out slightly to avoid API bursts
        try:
            for i, (pair_name, strategy) in enumerate(self.strategies.items()):
                # Small offset to prevent simultaneous API calls
                offset = i * 2  # 2 second offset between each
                
                if int(current_time - offset) % strategy.order_refresh_time == 0:
                    logger.debug(f"Executing {pair_name} strategy")
                    strategy.on_tick(current_time)
                    time.sleep(0.5)  # Small delay between strategies
                    
        except Exception as e:
            logger.error(f"Error in on_tick: {str(e)}", exc_info=True)
    
    # MODIFIED: Update did_fill_order to track daily PnL
    def did_fill_order(self, event):
        """FIXED: Track daily PnL and route to strategies"""
        try:
            # Update daily PnL
            trade_value = Decimal(str(event.amount)) * Decimal(str(event.price))
            fee = trade_value * Decimal("0.0026")
            
            if event.trade_type.name == "BUY":
                self.daily_pnl -= (trade_value + fee)
            else:
                self.daily_pnl += (trade_value - fee)
            
            logger.info(f"Daily PnL updated: ${float(self.daily_pnl):.2f} / ${float(self.max_daily_loss):.2f} limit")
            
            # Check daily loss limit
            if abs(self.daily_pnl) >= self.max_daily_loss:
                self.halt_until = time.time() + 86400
                logger.error(f"DAILY LOSS LIMIT REACHED: ${float(self.daily_pnl):.2f}. Trading halted for 24 hours.")
                
        except Exception as e:
            logger.error(f"Error updating PnL: {str(e)}")
            pass
        
        # Route to individual strategies
        for strategy in self.strategies.values():
            try:
                strategy.did_fill_order(event)
            except Exception as e:
                logger.error(f"Error routing fill event: {str(e)}")
                pass
    
    def format_status(self) -> str:
        """Aggregate status from all strategies"""
        status_lines = [
            "=" * 80,
            "MULTI-PAIR MARKET MAKING STATUS",
            "=" * 80
        ]
        
        # Get status from each strategy
        for pair_name, strategy in self.strategies.items():
            try:
                status = strategy.format_status()
                status_lines.append(status)
            except Exception:
                status_lines.append(f"{pair_name}: Error retrieving status")
        
        # Add summary
        status_lines.append("=" * 80)
        total_buys = sum(strategy.buy_trades for strategy in self.strategies.values())
        total_sells = sum(strategy.sell_trades for strategy in self.strategies.values())
        total_fees = sum(strategy.estimated_fees for strategy in self.strategies.values())
        status_lines.append(f"Total Trades: {total_buys} buys / {total_sells} sells | Fees: ${float(total_fees):.2f}")
        
        return "\n".join(status_lines)