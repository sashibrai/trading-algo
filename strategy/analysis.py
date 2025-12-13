from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AOI:
    """Represents an Area of Interest"""
    zone_high: float
    zone_low: float
    reactions_bullish: int = 0
    reactions_bearish: int = 0

    @property
    def total_reactions(self) -> int:
        return self.reactions_bullish + self.reactions_bearish

    @property
    def is_valid(self) -> bool:
        return self.total_reactions >= 3 and self.reactions_bullish >= 1 and self.reactions_bearish >= 1

    def is_price_in_zone(self, price: float) -> bool:
        return self.zone_low <= price <= self.zone_high

@dataclass
class Trendline:
    start_idx: datetime
    end_idx: datetime
    start_price: float
    end_price: float
    slope: float
    is_bullish: bool

    def get_value_at(self, idx_timestamp) -> float:
        if isinstance(idx_timestamp, pd.Timestamp):
             time_diff = (idx_timestamp - self.start_idx).total_seconds()
             return self.start_price + (self.slope * time_diff)
        return self.start_price

def resample_candles(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty: return df
    if not isinstance(df.index, pd.DatetimeIndex):
        try: df.index = pd.to_datetime(df['ts'], unit='s')
        except: pass
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    valid_agg = {k: v for k, v in agg_dict.items() if k in df.columns}
    resampled = df.resample(rule).agg(valid_agg).dropna()
    return resampled

def find_swing_points(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    df = df.copy()
    is_swing_high = pd.Series(True, index=df.index)
    is_swing_low = pd.Series(True, index=df.index)
    for i in range(1, left + 1):
        is_swing_high &= (df['high'] > df['high'].shift(i))
        is_swing_low &= (df['low'] < df['low'].shift(i))
    for i in range(1, right + 1):
        is_swing_high &= (df['high'] > df['high'].shift(-i))
        is_swing_low &= (df['low'] < df['low'].shift(-i))
    df['is_swing_high'] = is_swing_high
    df['is_swing_low'] = is_swing_low
    return df

def identify_daily_trend(df: pd.DataFrame, ema_period: int = 50) -> str:
    if len(df) < ema_period + 5: return "NEUTRAL"
    df = df.copy()
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    swings = find_swing_points(df, left=2, right=2)
    last_shs = swings[swings['is_swing_high']].tail(2)
    last_sls = swings[swings['is_swing_low']].tail(2)
    if len(last_shs) < 2 or len(last_sls) < 2: return "NEUTRAL"
    sh1, sh2 = last_shs.iloc[0]['high'], last_shs.iloc[1]['high']
    sl1, sl2 = last_sls.iloc[0]['low'], last_sls.iloc[1]['low']
    last_close = df.iloc[-1]['close']
    last_ema = df.iloc[-1]['ema']
    min_move = 0.001 * last_close
    valid_hh = (sh2 - sh1) > min_move
    valid_ll = (sl1 - sl2) > min_move
    structure_bullish = (sh2 > sh1) and (sl2 > sl1) and valid_hh
    structure_bearish = (sh2 < sh1) and (sl2 < sl1) and valid_ll
    if structure_bullish and last_close > last_ema: return "BULLISH"
    elif structure_bearish and last_close < last_ema: return "BEARISH"
    else: return "NEUTRAL"

def find_aois(df: pd.DataFrame, lookback: int = 200, zone_pct: float = 0.002, reaction_min_pct: float = 0.005) -> List[AOI]:
    df_subset = df.tail(lookback).copy()
    if df_subset.empty: return []
    swings = find_swing_points(df_subset, left=2, right=2)
    potential_levels = []
    potential_levels.extend(swings[swings['is_swing_high']]['high'].tolist())
    potential_levels.extend(swings[swings['is_swing_low']]['low'].tolist())
    potential_levels = sorted(list(set(potential_levels)))
    valid_aois = []
    for level in potential_levels:
        is_close = False
        for valid_aoi in valid_aois:
            if valid_aoi.is_price_in_zone(level):
                 is_close = True
                 break
        if is_close: continue
        zone_width = level * zone_pct
        zone_high = level + zone_width
        zone_low = level - zone_width
        bullish_reactions, bearish_reactions = 0, 0
        i = 0
        while i < len(df_subset):
            row = df_subset.iloc[i]
            touches = (row['high'] >= zone_low) and (row['low'] <= zone_high)
            if touches:
                bull_target = level * (1 + reaction_min_pct)
                bear_target = level * (1 - reaction_min_pct)
                broken_support, broken_resistance = False, False
                j, resolved = i, False
                while j < len(df_subset):
                    c_row = df_subset.iloc[j]
                    hit_bull = c_row['high'] >= bull_target
                    hit_bear = c_row['low'] <= bear_target
                    broken_support |= c_row['low'] < zone_low
                    broken_resistance |= c_row['high'] > zone_high
                    if hit_bull and not broken_support:
                        bullish_reactions += 1
                        resolved = True; break
                    if hit_bear and not broken_resistance:
                        bearish_reactions += 1
                        resolved = True; break
                    if broken_support and broken_resistance:
                        resolved = True; break
                    j += 1
                i = j + 1 if resolved else i + 1
            else: i += 1
        if (bullish_reactions + bearish_reactions) >= 3 and bullish_reactions >= 1 and bearish_reactions >= 1:
            valid_aois.append(AOI(zone_high, zone_low, bullish_reactions, bearish_reactions))
    return valid_aois

def find_trendlines(df: pd.DataFrame) -> List[Trendline]:
    swings = find_swing_points(df, left=2, right=2)
    trendlines = []
    sls = swings[swings['is_swing_low']]
    if len(sls) >= 2:
        sl_last = sls.iloc[-1]
        sl_prev = sls.iloc[-2]
        if sl_last['low'] > sl_prev['low']:
             t_diff = (sl_last.name - sl_prev.name).total_seconds()
             slope = (sl_last['low'] - sl_prev['low']) / t_diff
             trendlines.append(Trendline(start_idx=sl_prev.name, end_idx=sl_last.name, start_price=sl_prev['low'], end_price=sl_last['low'], slope=slope, is_bullish=True))
    shs = swings[swings['is_swing_high']]
    if len(shs) >= 2:
        sh_last = shs.iloc[-1]
        sh_prev = shs.iloc[-2]
        if sh_last['high'] < sh_prev['high']:
             t_diff = (sh_last.name - sh_prev.name).total_seconds()
             slope = (sh_last['high'] - sh_prev['high']) / t_diff
             trendlines.append(Trendline(start_idx=sh_prev.name, end_idx=sh_last.name, start_price=sh_prev['high'], end_price=sh_last['high'], slope=slope, is_bullish=False))
    return trendlines

def check_intraday_entry(df: pd.DataFrame, zone: AOI) -> Optional[str]:
    if len(df) < 5: return None
    current_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]

    recent_interaction = False
    for k in range(1, 4):
        c = df.iloc[-k]
        if zone.is_price_in_zone(c['high']) or zone.is_price_in_zone(c['low']):
            recent_interaction = True
            break
    if not recent_interaction: return None

    swings = find_swing_points(df, left=1, right=1)
    last_sh_idx = swings[swings['is_swing_high']].last_valid_index()
    last_sl_idx = swings[swings['is_swing_low']].last_valid_index()
    if last_sh_idx is None or last_sl_idx is None: return None

    if len(df) - df.index.get_loc(last_sh_idx) > 4: last_sh_idx = None
    if len(df) - df.index.get_loc(last_sl_idx) > 4: last_sl_idx = None

    if last_sh_idx is not None:
        sh2 = df.loc[last_sh_idx]
        prev_sh_idxs = [idx for idx in swings[swings['is_swing_high']].index if idx != last_sh_idx]
        if len(prev_sh_idxs) >= 1:
            sh1 = df.loc[prev_sh_idxs[-1]]
            if zone.is_price_in_zone(sh2['high']):
                if sh2['high'] < sh1['high']:
                    if current_candle['close'] < prev_candle['low']: return 'SELL'
                elif sh2['high'] > sh1['high']:
                    sh2_pos = df.index.get_loc(last_sh_idx)
                    if sh2_pos + 1 < len(df):
                        reversal = df.iloc[sh2_pos + 1]
                        if reversal['close'] < (sh2['high'] + sh2['low'])/2:
                             if current_candle['close'] < reversal['low']: return 'SELL'

    if last_sl_idx is not None:
        sl2 = df.loc[last_sl_idx]
        prev_sl_idxs = [idx for idx in swings[swings['is_swing_low']].index if idx != last_sl_idx]
        if len(prev_sl_idxs) >= 1:
            sl1 = df.loc[prev_sl_idxs[-1]]
            if zone.is_price_in_zone(sl2['low']):
                if sl2['low'] > sl1['low']:
                     if current_candle['close'] > prev_candle['high']: return 'BUY'
                elif sl2['low'] < sl1['low']:
                    sl2_pos = df.index.get_loc(last_sl_idx)
                    if sl2_pos + 1 < len(df):
                        reversal = df.iloc[sl2_pos + 1]
                        if reversal['close'] > (sl2['high'] + sl2['low'])/2:
                            if current_candle['close'] > reversal['high']: return 'BUY'
    return None
