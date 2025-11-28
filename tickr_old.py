"""
Tickr.py
Standalone implementation for Hummingbot (no external dependencies on tickr package)

This module provides two ways to fetch candle data:
1. Direct from exchange using ccxt (recommended for Hummingbot)
2. From GitHub repository (original tickr functionality)

License: MIT
Author: syncsoftco (modified for Hummingbot)
"""

import ccxt
import pandas as pd
import numpy as np
from typing import Optional, List
import logging
import time
import datetime

logger = logging.getLogger(__name__)


class TickrClient:
    """
    Simplified Tickr client for Hummingbot
    Fetches OHLCV data directly from exchanges using ccxt
    
    This is a lightweight alternative to the full GitHub-based tickr package
    that works reliably in Hummingbot's environment.
    
    Usage:
        client = TickrClient(exchange="kraken")
        df = client.get_candles("BTC/USDT", "15m", limit=200)
    """
    
    def __init__(self, exchange: str = "kucoin"):
        """
        Initialize TickrClient with exchange
        
        Args:
            exchange: Exchange name ('kraken', 'kucoin', 'coinbase', etc.)
        """
        self.exchange_name = exchange.lower()
        
        # Map exchange names to ccxt classes
        exchange_map = {
            'kraken': ccxt.kraken,
            'kucoin': ccxt.kucoin,
            'coinbase': ccxt.coinbase,
            'bybit': ccxt.bybit,
            'okx': ccxt.okx,
            'huobi': ccxt.huobi,
            'gateio': ccxt.gateio,
        }
        
        exchange_class = exchange_map.get(self.exchange_name)
        if not exchange_class:
            logger.warning(f"Exchange '{exchange}' not in map, trying ccxt.{exchange}")
            try:
                exchange_class = getattr(ccxt, self.exchange_name)
            except AttributeError:
                raise ValueError(f"Exchange '{exchange}' not supported by ccxt")
        
        # Initialize exchange with settings optimized for reliability
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            },
            'timeout': 30000,
            'rateLimit': 1200,
        })
        
        logger.info(f"TickrClient initialized for {self.exchange_name.upper()}")
    
    def get_candles(
        self, 
        symbol: str, 
        timeframe: str = '15m', 
        limit: int = 200,
        since: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV candles from exchange
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT', 'ATOM/USDT', 'ATOM-USDT')
            timeframe: Candle interval - '1m', '5m', '15m', '1h', '4h', '1d'
            limit: Number of candles to fetch (max varies by exchange)
            since: Timestamp in milliseconds to fetch from (optional)
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: timestamp (datetime, UTC)
            Returns None on error
        
        Example:
            >>> client = TickrClient(exchange="kraken")
            >>> df = client.get_candles("BTC/USDT", "1h", limit=100)
            >>> print(df.tail())
        """
        try:
            # Normalize symbol format (handle both BTC/USDT and BTC-USDT)
            symbol_formatted = symbol.replace('-', '/')
            
            # Validate timeframe
            valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w']
            if timeframe not in valid_timeframes:
                logger.warning(f"Timeframe '{timeframe}' may not be supported. Valid: {valid_timeframes}")
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol_formatted,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            
            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"{self.exchange_name.upper()}: No candle data for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime (UTC)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            # Ensure numeric types for all OHLCV columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            if len(df) == 0:
                logger.warning(f"{self.exchange_name.upper()}: All data was NaN for {symbol}")
                return None
            
            logger.debug(
                f"{self.exchange_name.upper()}: Fetched {len(df)} {timeframe} candles for {symbol} "
                f"({df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')})"
            )
            
            return df
            
        except ccxt.NetworkError as e:
            logger.error(f"{self.exchange_name.upper()}: Network error fetching {symbol}: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"{self.exchange_name.upper()}: Exchange error fetching {symbol}: {e}")
            return None
        except ccxt.BaseError as e:
            logger.error(f"{self.exchange_name.upper()}: CCXT error fetching {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(
                f"{self.exchange_name.upper()}: Unexpected error fetching {symbol}: {e}",
                exc_info=True
            )
            return None
    
    def get_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime.datetime,
        end_time: Optional[datetime.datetime] = None,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical candles between two timestamps with automatic pagination
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle interval
            start_time: Start datetime (UTC)
            end_time: End datetime (UTC), defaults to now
            max_retries: Number of retry attempts per request
        
        Returns:
            DataFrame with all candles in the time range
        """
        if end_time is None:
            end_time = datetime.datetime.now(datetime.timezone.utc)
        
        # Convert to milliseconds
        since = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        all_candles = []
        current_since = since
        
        logger.info(
            f"Fetching historical {timeframe} candles for {symbol} "
            f"from {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}"
        )
        
        while current_since < end_ts:
            for attempt in range(max_retries):
                try:
                    df = self.get_candles(symbol, timeframe, limit=1000, since=current_since)
                    
                    if df is None or len(df) == 0:
                        logger.warning(f"No more data available at {current_since}")
                        return self._combine_candles(all_candles) if all_candles else None
                    
                    all_candles.append(df)
                    
                    # Move to next batch
                    last_timestamp = df.index[-1].timestamp() * 1000
                    current_since = int(last_timestamp + 1)
                    
                    # Rate limiting
                    time.sleep(self.exchange.rateLimit / 1000)
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                        time.sleep(2 ** attempt)
                    else:
                        logger.error(f"Failed after {max_retries} attempts")
                        return self._combine_candles(all_candles) if all_candles else None
        
        return self._combine_candles(all_candles)
    
    def _combine_candles(self, candle_dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Combine multiple candle DataFrames and remove duplicates"""
        if not candle_dfs:
            return None
        
        combined = pd.concat(candle_dfs)
        # Remove duplicates keeping first occurrence
        combined = combined[~combined.index.duplicated(keep='first')]
        combined = combined.sort_index()
        
        return combined
    
    def get_markets(self) -> List[str]:
        """
        Get list of available trading pairs on the exchange
        
        Returns:
            List of symbol strings (e.g., ['BTC/USDT', 'ETH/USDT', ...])
        """
        try:
            self.exchange.load_markets()
            return list(self.exchange.markets.keys())
        except Exception as e:
            logger.error(f"Error loading markets: {e}")
            return []
    
    def get_timeframes(self) -> List[str]:
        """
        Get list of supported timeframes for the exchange
        
        Returns:
            List of timeframe strings (e.g., ['1m', '5m', '1h', ...])
        """
        try:
            if hasattr(self.exchange, 'timeframes'):
                return list(self.exchange.timeframes.keys())
            return ['1m', '5m', '15m', '1h', '4h', '1d']  # Common defaults
        except Exception as e:
            logger.error(f"Error getting timeframes: {e}")
            return []


# Expose for easy import
__all__ = ['TickrClient']


# Self-test when run directly
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("Tickr Self-Test")
    print("=" * 60)
    
    # Test with Kucoin
    try:
        print("\nTesting Kucoin...")
        client = TickrClient(exchange="kucoin")
        
        # Test basic fetch
        df = client.get_candles("BTC/USD", "1h", limit=5)
        if df is not None:
            print(f"✅ SUCCESS: Fetched {len(df)} candles")
            print(df[['open', 'high', 'low', 'close', 'volume']])
        else:
            print("❌ FAILED: No data returned")
        
        # Test available markets
        markets = client.get_markets()
        print(f"\n✅ Found {len(markets)} trading pairs")
        print(f"Sample pairs: {markets[:5]}")
        
        # Test timeframes
        timeframes = client.get_timeframes()
        print(f"\n✅ Supported timeframes: {timeframes}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

