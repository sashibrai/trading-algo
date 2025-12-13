import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import timedelta
from strategy.analysis import resample_candles, identify_daily_trend, find_aois, check_intraday_entry, AOI
from tqdm import tqdm

def load_data(filepath):
    """
    Load 1-minute data from CSV.
    Expected columns: date, open, high, low, close
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        # Parse date
        # auto-detect format?
        df['ts'] = pd.to_datetime(df['date'], dayfirst=True)
        df.set_index('ts', inplace=True)

        # Ensure numeric
        cols = ['open', 'high', 'low', 'close']
        for c in cols:
            df[c] = pd.to_numeric(df[c])

        if 'volume' not in df.columns:
            df['volume'] = 0

        return df.sort_index()
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def backtest(data_path, config=None):
    # 1. Load Data
    df_1m = load_data(data_path)

    # 2. Resample
    print("Resampling data...")
    df_5m = resample_candles(df_1m, '5min')
    df_4h = resample_candles(df_1m, '4h')
    df_1d = resample_candles(df_1m, '1D')

    print(f"Data Points: 5m={len(df_5m)}, 4H={len(df_4h)}, 1D={len(df_1d)}")

    # 3. Pre-calculate Daily Trends (to avoid recalculating every step)
    # We map Date -> Trend.
    # Note: Trend for Day T is calculated using data up to T-1.
    print("Pre-calculating Daily Trends...")
    daily_trends = {} # Date -> Trend
    # We need a rolling window or expanding window.
    # To be strictly causal: At timestamp T, we use Daily candles closed before T.

    # A simple way: Calculate Trend for every day using expanding window? Slow.
    # Rolling window?
    # `identify_daily_trend` takes a DF.
    # Let's compute it for each day.

    # For simulation speed, let's optimize:
    # We can iterate daily candles.
    trends_series = []

    # Need at least 50 days for EMA
    for i in range(50, len(df_1d)):
        # Data available *up to* yesterday (i-1) is used for Today (i)
        # Wait, if we are trading on Day X, we know Day X-1 Close.
        subset = df_1d.iloc[:i] # 0 to i-1
        trend = identify_daily_trend(subset)
        date = df_1d.index[i].date() # The date valid for 'today'
        daily_trends[date] = trend

    # 4. Simulation Loop
    # We iterate 5m candles.

    trades = []
    open_position = None # {type, entry_price, entry_time, sl, tp}

    current_aois = []
    last_aoi_update = None

    # Config Params
    quantity = config.get('quantity', 50) if config else 50
    # Stop Loss / Target logic handled dynamically in strategy usually,
    # but here we hardcode 1:3 or use the strategy's calculated SL?
    # Strategy calculates SL based on Swing.

    print("Starting Simulation...")
    # Need warm-up period for AOIs (200 4H candles)
    # 200 * 4 hours = 800 hours ~ 1.5 months.

    # Find start time
    if len(df_4h) < 200:
        print("Not enough data for AOI warmup (need 200 4H candles).")
        return

    start_idx_4h = 200
    warmup_end_time = df_4h.index[start_idx_4h]

    # Filter 5m data to start after warmup
    sim_data = df_5m[df_5m.index >= warmup_end_time]

    pbar = tqdm(total=len(sim_data))

    for time, candle in sim_data.iterrows():
        pbar.update(1)

        # A. Update AOIs if 4H candle closed
        # We need data up to 'time'.
        # Actually, AOIs are based on *closed* 4H candles.
        # Check if we have new 4H candles since last update.
        # Efficient way: Look up valid 4H candles before 'time'.

        # Optimization: Update AOIs only when 'time' crosses a 4H boundary.
        # But `find_aois` is slow.
        # Let's just update every 4 hours.

        should_update_aoi = False
        if last_aoi_update is None:
            should_update_aoi = True
        else:
            # If current time is > last_update + 4h
            if time - last_aoi_update >= timedelta(hours=4):
                should_update_aoi = True

        if should_update_aoi:
            # Get 4H data up to this time
            # mask: df_4h.index < time
            # We assume df_4h index is the *start* of the candle?
            # If start, then we can use candles where index + 4H <= time.
            # Usually resample labels left (start).
            # So candle at 10:00 closes at 14:00. At 14:00 we can use it.

            subset_4h = df_4h[df_4h.index < (time - timedelta(hours=4))]
            # This might be too aggressive filtering.
            # Let's say we are at 14:05. The 10:00-14:00 candle is closed.
            # Its label is 10:00. 10:00 < 14:05 - 4h (10:05). Yes.

            # Use fixed window (lookback 200)
            subset_4h = subset_4h.tail(200)

            if len(subset_4h) >= 50: # Min require
                current_aois = find_aois(subset_4h)
                last_aoi_update = time

        # B. Get Daily Trend
        current_date = time.date()
        daily_trend = daily_trends.get(current_date, "NEUTRAL")

        # C. Manage Open Position
        if open_position:
            # Check Exit
            # 1. SL Hit
            # 2. Target Hit (Optional, usually 1:2 or 1:3)
            # Strategy didn't strictly specify target, user said "Best entries... big moves".
            # Let's implement a Trailing SL or fixed Target?
            # User: "Stoploss of last high or low".
            # Let's assume Fixed R:R of 1:3 for backtest? Or just SL?
            # User didn't specify Target logic explicitly in prompt, other than "Buy at HL...".
            # Let's use 1:3 R:R for simulation to show "Gained Points".

            entry_price = open_position['entry_price']
            sl_price = open_position['sl']
            risk = abs(entry_price - sl_price)
            target_price = entry_price + (3 * risk) if open_position['type'] == 'BUY' else entry_price - (3 * risk)

            exit_reason = None
            exit_price = 0

            if open_position['type'] == 'BUY':
                if candle['low'] <= sl_price:
                    exit_reason = 'SL'
                    exit_price = sl_price # Assume slippage? Use SL price
                elif candle['high'] >= target_price:
                    exit_reason = 'Target'
                    exit_price = target_price
            else: # SELL
                if candle['high'] >= sl_price:
                    exit_reason = 'SL'
                    exit_price = sl_price
                elif candle['low'] <= target_price:
                    exit_reason = 'Target'
                    exit_price = target_price

            if exit_reason:
                points = exit_price - entry_price if open_position['type'] == 'BUY' else entry_price - exit_price
                trades.append({
                    'entry_time': open_position['entry_time'],
                    'exit_time': time,
                    'type': open_position['type'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'points': points,
                    'reason': exit_reason
                })
                open_position = None
                continue # Don't re-enter same bar

        # D. Check Entry
        if not open_position:
            # Filter by Trend
            # "Intraday entries ... aligned with HTF bias"

            # Check AOI interaction
            in_aoi = False
            active_aoi = None
            for aoi in current_aois:
                if aoi.is_price_in_zone(candle['high']) or aoi.is_price_in_zone(candle['low']):
                    in_aoi = True
                    active_aoi = aoi
                    break

            if in_aoi:
                # Need recent history for 5m structure check
                # Strategy uses df passed to check_intraday_entry
                # We need the simulation history up to now.
                # Window: Last ~20 candles is enough for swing detection?
                # `check_intraday_entry` needs enough to find swings.
                # `find_swing_points` uses +/- 2. Need at least 10-15.

                # Slicing is slow inside loop.
                # Try to keep a rolling buffer? Or just slice tail.

                # Get index of current candle
                # Slice last 50 candles
                # We can use `sim_data` up to `time`.
                # sim_data is 5m.
                # Warning: sim_data is huge.
                # Let's slice by integer location since we iterate.

                # Actually, `time` is index.
                # historical_5m = sim_data.loc[:time].tail(20) # This includes current candle 'time' as last?
                # Yes, in backtest 'candle' is the closed candle at 'time'.

                # Optimization: Access by integer position
                idx_pos = sim_data.index.get_loc(time)
                if idx_pos > 20:
                     window_5m = sim_data.iloc[idx_pos-20 : idx_pos+1]

                     signal = check_intraday_entry(window_5m, active_aoi)

                     if signal:
                         # Trend Filter
                         valid_trade = False
                         if signal == 'BUY' and daily_trend in ['BULLISH', 'NEUTRAL']:
                             valid_trade = True
                         elif signal == 'SELL' and daily_trend in ['BEARISH', 'NEUTRAL']:
                             valid_trade = True

                         if valid_trade:
                             # Calculate SL
                             # Use simple logic from strategy: Recent High/Low
                             recent = window_5m.tail(5)
                             if signal == 'BUY':
                                 sl = recent['low'].min()
                                 if sl >= candle['close']: sl = candle['close'] * 0.995
                             else:
                                 sl = recent['high'].max()
                                 if sl <= candle['close']: sl = candle['close'] * 1.005

                             open_position = {
                                 'type': signal,
                                 'entry_price': candle['close'],
                                 'entry_time': time,
                                 'sl': sl
                             }

    pbar.close()

    # Save Results
    results_df = pd.DataFrame(trades)
    if not results_df.empty:
        results_df.to_csv("backtest_results.csv", index=False)
        print("\nBacktest Complete.")
        print(f"Total Trades: {len(results_df)}")
        print(f"Total Points: {results_df['points'].sum():.2f}")
        print(f"Win Rate: {(len(results_df[results_df['points'] > 0]) / len(results_df) * 100):.1f}%")
        print("Detailed log saved to backtest_results.csv")
    else:
        print("\nBacktest Complete. No trades found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="Path to 1-minute OHLC CSV file")
    args = parser.parse_args()

    backtest(args.file)
