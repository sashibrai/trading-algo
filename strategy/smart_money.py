import os
import sys
import yaml
import time
import pandas as pd
from datetime import datetime, timedelta
from logger import logger
from brokers import BrokerGateway, OrderRequest, Exchange, OrderType, TransactionType, ProductType, Validity
from strategy.analysis import resample_candles, identify_daily_trend, find_aois, check_intraday_entry, AOI

class SmartMoneyStrategy:
    """
    Implementation of Smart Money Concepts Strategy:
    - 1W/1D Trend Analysis
    - 4H Area of Interest (AOI) Detection
    - 5m Intraday Entry Triggers (CHoCH, Fakeouts)
    """

    def __init__(self, broker: BrokerGateway, config: dict):
        self.broker = broker
        self.config = config

        # Parse Config
        self.symbol = config.get('symbol', 'NSE:NIFTY 50')
        self.trade_symbol = config.get('trade_symbol') # e.g. Future
        self.quantity = config.get('quantity', 50)

        # State
        self.daily_trend = "NEUTRAL"
        self.aois: List[AOI] = []
        self.open_position = None # Track current position to avoid spamming

        # Data Buffers
        self.ticks_buffer = []
        self.candles_5m = pd.DataFrame()

        # Initialize
        self._initialize_data()

    def _initialize_data(self):
        """Fetch historical data and build initial analysis."""
        logger.info("Initializing Smart Money Strategy...")

        end_date = datetime.now().strftime("%Y-%m-%d")
        # Need ~200 4H candles for AOI. 200 * 4 = 800 hours ~ 130 trading days.
        # Fetch 1 year of data to be safe for Daily trend + 4H AOI.
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        logger.info(f"Fetching history for {self.symbol} from {start_date} to {end_date}")

        # Fetch Daily Data
        try:
            # Fetch 1D for Trend
            daily_data = self.broker.get_history(self.symbol, "day", start_date, end_date)
            df_daily = pd.DataFrame(daily_data)
            if not df_daily.empty:
                df_daily['ts'] = pd.to_datetime(df_daily['ts'], unit='s')
                df_daily.set_index('ts', inplace=True)

                self.daily_trend = identify_daily_trend(df_daily)
                logger.info(f"Daily Trend: {self.daily_trend}")
            else:
                logger.warning("No daily data found!")

            # Fetch 60m for AOI (Resample to 4H)
            hourly_data = self.broker.get_history(self.symbol, "60m", start_date, end_date)
            df_hourly = pd.DataFrame(hourly_data)

            if not df_hourly.empty:
                df_hourly['ts'] = pd.to_datetime(df_hourly['ts'], unit='s')
                df_hourly.set_index('ts', inplace=True)

                df_4h = resample_candles(df_hourly, '4H')
                logger.info(f"Generated {len(df_4h)} 4H candles")

                self.aois = find_aois(df_4h,
                                      lookback=self.config.get('aoi_lookback', 200),
                                      zone_pct=self.config.get('aoi_zone_pct', 0.002),
                                      reaction_min_pct=self.config.get('reaction_min_pct', 0.005))
                logger.info(f"Found {len(self.aois)} valid AOIs")
                for aoi in self.aois:
                    logger.info(f"AOI: {aoi.zone_low} - {aoi.zone_high} (Reactions: {aoi.total_reactions})")
            else:
                logger.warning("No hourly data found!")

            # Fetch recent 5m data for context (last 2 days)
            start_5m = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            data_5m = self.broker.get_history(self.symbol, "5m", start_5m, end_date)
            df_5m = pd.DataFrame(data_5m)
            if not df_5m.empty:
                df_5m['ts'] = pd.to_datetime(df_5m['ts'], unit='s')
                df_5m.set_index('ts', inplace=True)
                self.candles_5m = df_5m
                logger.info(f"Loaded {len(self.candles_5m)} recent 5m candles")

            # Initialize position state from broker
            positions = self.broker.get_positions()
            for p in positions:
                if p.symbol == self.config.get('trade_symbol', self.symbol) and p.quantity_total != 0:
                    self.open_position = p
                    logger.info(f"Found existing position: {p}")
                    break

        except Exception as e:
            logger.error(f"Error initializing data: {e}", exc_info=True)

    def on_ticks_update(self, tick_data):
        """
        Handle real-time ticks.
        1. Accumulate ticks to form 5m candles.
        2. On candle close, run entry logic.
        """
        if isinstance(tick_data, list):
             for t in tick_data:
                 self._process_tick(t)
        else:
             self._process_tick(tick_data)

    def _process_tick(self, tick):
        price = tick.get('last_price') or tick.get('ltp')
        if not price:
            return

        ts = datetime.now()
        current_5m_start = ts.replace(second=0, microsecond=0, minute=(ts.minute // 5) * 5)

        if self.candles_5m.empty:
            self._start_new_candle(current_5m_start, price)
            return

        last_candle_time = self.candles_5m.index[-1]

        if current_5m_start > last_candle_time:
            self._on_candle_close(self.candles_5m.iloc[-1])
            self._start_new_candle(current_5m_start, price)
        else:
            self._update_current_candle(price)

    def _start_new_candle(self, ts, price):
        new_row = pd.DataFrame({
            'open': [price], 'high': [price], 'low': [price], 'close': [price], 'volume': [0]
        }, index=[ts])
        self.candles_5m = pd.concat([self.candles_5m, new_row])

    def _update_current_candle(self, price):
        idx = self.candles_5m.index[-1]
        self.candles_5m.at[idx, 'high'] = max(self.candles_5m.at[idx, 'high'], price)
        self.candles_5m.at[idx, 'low'] = min(self.candles_5m.at[idx, 'low'], price)
        self.candles_5m.at[idx, 'close'] = price

    def _on_candle_close(self, candle):
        logger.info(f"5m Candle Closed: {candle.name} O:{candle['open']} H:{candle['high']} L:{candle['low']} C:{candle['close']}")

        # Position safety check: Don't scan for entry if already in position
        if self.open_position:
            return

        # 1. Check if price is in any AOI
        in_aoi = False
        active_aoi = None

        current_high = candle['high']
        current_low = candle['low']

        for aoi in self.aois:
            if aoi.is_price_in_zone(current_high) or aoi.is_price_in_zone(current_low):
                in_aoi = True
                active_aoi = aoi
                break

        if not in_aoi:
            return

        logger.info(f"Price inside AOI: {active_aoi.zone_low} - {active_aoi.zone_high}")

        # 2. Check Entry Signal
        signal = check_intraday_entry(self.candles_5m, active_aoi)

        if signal:
            logger.info(f"ENTRY SIGNAL: {signal}")

            allow_trade = False
            if signal == 'BUY':
                if self.daily_trend in ['BULLISH', 'NEUTRAL']:
                    allow_trade = True
                else:
                    logger.info("Skipping BUY signal due to BEARISH Daily Trend")

            elif signal == 'SELL':
                if self.daily_trend in ['BEARISH', 'NEUTRAL']:
                    allow_trade = True
                else:
                    logger.info("Skipping SELL signal due to BULLISH Daily Trend")

            if allow_trade:
                self._execute_trade(signal, candle['close'])

    def _calculate_stop_loss(self, signal, close_price):
        """
        Calculate stop loss based on recent structure.
        User req: Stoploss of last high or low.
        """
        # We look at the last few candles (entry trigger sequence) to find the local extreme.
        # Since we just confirmed a swing and reversal, the extreme should be nearby.
        recent_window = self.candles_5m.tail(5)

        if signal == 'BUY':
             # Stop below the recent low
             stop_price = recent_window['low'].min()
             # Fallback if too close
             if stop_price > close_price * 0.999: # 0.1% min distance
                  stop_price = close_price * (1 - self.config.get('stop_loss_pct', 0.005))
        else: # SELL
             stop_price = recent_window['high'].max()
             if stop_price < close_price * 1.001:
                  stop_price = close_price * (1 + self.config.get('stop_loss_pct', 0.005))

        return stop_price

    def _execute_trade(self, signal, current_price):
        # Double check safety
        if self.open_position:
            logger.warning("Attempted to trade while position open. Skipping.")
            return

        qty = self.config.get('quantity', 1)
        symbol = self.config.get('trade_symbol', self.symbol)

        txn_type = TransactionType.BUY if signal == 'BUY' else TransactionType.SELL

        # Calculate Stop Loss
        stop_price = self._calculate_stop_loss(signal, current_price)

        logger.info(f"Placing {signal} Order for {symbol} Qty {qty} SL: {stop_price}")

        req = OrderRequest(
            symbol=symbol,
            exchange=Exchange[self.config.get('exchange', 'NFO')],
            transaction_type=txn_type,
            quantity=qty,
            product_type=ProductType.MARGIN,
            order_type=OrderType.MARKET,
            validity=Validity.DAY,
            tag="SmartMoney",
            # Pass SL in extras or try to place SL-M if supported (requires separate order usually)
            extras={"stop_loss": stop_price}
        )

        try:
            resp = self.broker.place_order(req)
            logger.info(f"Order Placed: {resp}")

            self.open_position = "PENDING"

            # Place SL Order (If broker supports it, otherwise we need a manager)
            # Zerodha/Paper support SL-M.
            # We place a counter order with STOP type.

            sl_txn_type = TransactionType.SELL if signal == 'BUY' else TransactionType.BUY

            sl_req = OrderRequest(
                symbol=symbol,
                exchange=Exchange[self.config.get('exchange', 'NFO')],
                transaction_type=sl_txn_type,
                quantity=qty,
                product_type=ProductType.MARGIN,
                order_type=OrderType.STOP, # SL-M
                stop_price=stop_price,
                validity=Validity.DAY,
                tag="SmartMoney_SL"
            )

            # Sleep briefly to ensure primary order is processed?
            # In async world or fast market, might need confirm.
            # For now, fire and forget SL order.
            try:
                sl_resp = self.broker.place_order(sl_req)
                logger.info(f"SL Order Placed: {sl_resp}")
            except Exception as sl_e:
                 logger.error(f"Failed to place SL Order: {sl_e}")

        except Exception as e:
            logger.error(f"Order Placement Failed: {e}")

if __name__ == "__main__":
    from queue import Queue
    from dispatcher import DataDispatcher

    config_path = os.path.join(os.path.dirname(__file__), "configs/smart_money.yml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['default']

    broker_name = os.getenv("BROKER_NAME", "paper")
    broker = BrokerGateway.from_name(broker_name)

    dispatcher = DataDispatcher()
    queue = Queue()
    dispatcher.register_main_queue(queue)

    strategy = SmartMoneyStrategy(broker, config)

    def on_ticks(ws, ticks):
        if isinstance(ticks, list):
            dispatcher.dispatch(ticks)
        else:
            if "symbol" in ticks or "last_price" in ticks:
                dispatcher.dispatch(ticks)

    broker.connect_websocket(on_ticks=on_ticks)
    broker.symbols_to_subscribe([config['symbol']])

    logger.info("Starting Strategy Loop...")

    try:
        while True:
            data = queue.get()
            strategy.on_ticks_update(data)
    except KeyboardInterrupt:
        logger.info("Stopping...")
