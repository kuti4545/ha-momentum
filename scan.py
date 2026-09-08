#!/usr/bin/env python3
"""
TV 15m strategy port.
15m x 24 = 360dk HTF. Her kapanmış 15m'de (lookahead) 6H HA EMA20 %mom.
Sinyal: son iki 15m kapanışında mom 0 kesişimi — TV plotshape ile aynı an.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ha-momentum/1.1"})
TR = timezone(timedelta(hours=3))
SIX_H = 6 * 60 * 60 * 1000
FIFTEEN = 15 * 60 * 1000


def now_tr() -> str:
    return datetime.now(TR).strftime("%Y-%m-%d %H:%M TR")


def get_tickers() -> list[dict]:
    r = SESSION.get(
        f"{config.BITGET_BASE}/api/v2/mix/market/tickers",
        params={"productType": config.PRODUCT_TYPE},
        timeout=20,
    )
    r.raise_for_status()
    rows = []
    for row in r.json().get("data") or []:
        sym = row.get("symbol") or ""
        if sym in config.SKIP:
            continue
        try:
            vol = float(row.get("usdtVolume") or 0)
        except (TypeError, ValueError):
            vol = 0.0
        if vol < config.MIN_VOLUME:
            continue
        rows.append(row)
    rows.sort(key=lambda x: float(x.get("usdtVolume") or 0), reverse=True)
    return rows[: config.MAX_SYMBOLS]


def fetch_candles(symbol: str, gran: str, limit: int) -> pd.DataFrame | None:
    try:
        r = SESSION.get(
            f"{config.BITGET_BASE}/api/v2/mix/market/candles",
            params={
                "symbol": symbol,
                "granularity": gran,
                "limit": str(limit),
                "productType": config.PRODUCT_TYPE,
            },
            timeout=15,
        )
        raw = (r.json().get("data") or [])
        if len(raw) < 20:
            return None
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "base_vol", "quote_vol"])
        for col in ["ts", "open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna().sort_values("ts").reset_index(drop=True)
    except requests.RequestException:
        return None


def drop_unclosed_15m(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    last = int(df["ts"].iloc[-1])
    now_ms = int(time.time() * 1000)
    if now_ms < last + FIFTEEN - 3000:
        return df.iloc[:-1].reset_index(drop=True)
    return df


def heikin_ashi_close(o, h, l, c) -> np.ndarray:
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty_like(ha_c)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(c)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    return ha_c


def ema(arr: np.ndarray, n: int) -> np.ndarray:
    k = 2.0 / (n + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


def momentum_of(df6: pd.DataFrame) -> float | None:
    if df6 is None or len(df6) < config.MA_LEN + 2:
        return None
    o = df6["open"].to_numpy(dtype=float)
    h = df6["high"].to_numpy(dtype=float)
    l = df6["low"].to_numpy(dtype=float)
    c = df6["close"].to_numpy(dtype=float)
    ha_c = heikin_ashi_close(o, h, l, c)
    ma = ema(ha_c, config.MA_LEN)
    if ma[-2] == 0:
        return None
    return float((ma[-1] - ma[-2]) / ma[-2] * 100.0)


def merge_forming(df6: pd.DataFrame, df15: pd.DataFrame, drop_last_15: bool) -> pd.DataFrame:
    out = df6.copy()
    if df15 is None or df15.empty:
        return out
    work = df15.iloc[:-1] if drop_last_15 and len(df15) else df15
    if work.empty:
        return out
    start = int(out["ts"].iloc[-1])
    end = start + SIX_H
    piece = work[(work["ts"] >= start) & (work["ts"] < end)]
    if piece.empty:
        return out
    out.loc[out.index[-1], "high"] = max(float(out["high"].iloc[-1]), float(piece["high"].max()))
    out.loc[out.index[-1], "low"] = min(float(out["low"].iloc[-1]), float(piece["low"].min()))
    out.loc[out.index[-1], "close"] = float(piece["close"].iloc[-1])
    return out


def eval_symbol(symbol: str) -> dict | None:
    d6 = fetch_candles(symbol, "6H", 80)
    d15 = fetch_candles(symbol, "15m", 40)
    time.sleep(0.05)
    if d6 is None or d15 is None:
        return None
    d15 = drop_unclosed_15m(d15)
    if d15 is None or len(d15) < 3:
        return None
    now_df = merge_forming(d6, d15, drop_last_15=False)
    prev_df = merge_forming(d6, d15, drop_last_15=True)
    m1 = momentum_of(now_df)
    m0 = momentum_of(prev_df)
    if m0 is None or m1 is None:
        return None
    side = None
    if m0 <= 0 < m1:
        side = "LONG"
    elif m0 >= 0 > m1:
        side = "SHORT"
    return {
        "side": side,
        "m0": round(m0, 4),
        "m1": round(m1, 4),
        "close": float(d15["close"].iloc[-1]),
    }


def load_state() -> dict:
    p = Path(config.STATE_PATH)
    if not p.exists():
        return {"last": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"last": {}}


def save_state(state: dict) -> None:
    p = Path(config.STATE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def cooldown_ok(state: dict, key: str) -> bool:
    last = (state.get("last") or {}).get(key)
    if not last:
        return True
    try:
        ts = datetime.fromisoformat(last)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    return datetime.now(timezone.utc) - ts >= timedelta(minutes=config.COOLDOWN_MIN)


def send_tg(text: str) -> bool:
    if not (config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID):
        print(text)
        return False
    try:
        r = SESSION.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        return r.status_code == 200
    except requests.RequestException as exc:
        print("tg", exc)
        return False


def fmt(sym: str, side: str, info: dict) -> str:
    arrow = "🟢 LONG" if side == "LONG" else "🔴 SHORT"
    return (
        f"{arrow}  <b>{sym}</b>  [HA-MOM 15m]\n"
        f"{now_tr()}\n"
        f"Fiyat: <b>{info['close']}</b>\n"
        f"15m mom: {info['m0']} → {info['m1']}\n"
        f"TV: 15m + aynı strategy, son kapanmış 15m ok.\n"
        f"<i>Tavsiye değildir.</i>"
    )


def main() -> None:
    tickers = get_tickers()
    print(f"{len(tickers)} kripto")
    state = load_state()
    sent = 0
    for row in tickers:
        sym = row.get("symbol")
        try:
            info = eval_symbol(sym)
        except Exception as exc:
            print("err", sym, exc)
            continue
        if not info or not info["side"]:
            continue
        side = info["side"]
        key = f"{sym}:{side}"
        if not cooldown_ok(state, key):
            print("cd", key)
            continue
        if send_tg(fmt(sym, side, info)):
            state.setdefault("last", {})[key] = datetime.now(timezone.utc).isoformat()
            sent += 1
            print("ok", key, info)
        if sent >= 6:
            break
    save_state(state)
    print("bitti", sent)


if __name__ == "__main__":
    main()
