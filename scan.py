#!/usr/bin/env python3
"""
TV: 15m grafikte 12H-8H MA Momentum Strategy
15m x 24 = 360dk = 6H Heikin Ashi close -> EMA20 -> % momentum 0 kesişimi.
lookahead_on = kapanmamis 6H mum da var (TV ile ayni).
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
SESSION.headers.update({"User-Agent": "ha-momentum/1.0"})
TR = timezone(timedelta(hours=3))


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


def candles_6h(symbol: str, limit: int = 80) -> pd.DataFrame | None:
    try:
        r = SESSION.get(
            f"{config.BITGET_BASE}/api/v2/mix/market/candles",
            params={
                "symbol": symbol,
                "granularity": "6H",
                "limit": str(limit),
                "productType": config.PRODUCT_TYPE,
            },
            timeout=15,
        )
        body = r.json()
        raw = body.get("data") or []
        if len(raw) < 30:
            return None
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "base_vol", "quote_vol"])
        for col in ["ts", "open", "high", "low", "close", "base_vol", "quote_vol"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().sort_values("ts").reset_index(drop=True)
        return df if len(df) >= 30 else None
    except requests.RequestException:
        return None


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty_like(ha_c)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(c)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    out = df.copy()
    out["ha_close"] = ha_c
    out["ha_open"] = ha_o
    return out


def ema(arr: np.ndarray, n: int) -> np.ndarray:
    k = 2.0 / (n + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


def signal_from_6h(df: pd.DataFrame) -> dict | None:
    ha = heikin_ashi(df)
    ma = ema(ha["ha_close"].to_numpy(dtype=float), config.MA_LEN)
    if len(ma) < config.MA_LEN + 3:
        return None
    # lookback 1, percent, smooth 1
    prev = ma[:-1]
    mom = np.zeros_like(ma)
    mom[1:] = np.where(prev != 0, (ma[1:] - prev) / prev * 100.0, 0.0)
    m0, m1 = float(mom[-2]), float(mom[-1])
    side = None
    if m0 <= 0 < m1:
        side = "LONG"
    elif m0 >= 0 > m1:
        side = "SHORT"
    return {
        "side": side,
        "momentum": round(m1, 4),
        "momentum_prev": round(m0, 4),
        "ma": float(ma[-1]),
        "ha_close": float(ha["ha_close"].iloc[-1]),
        "bar_close": float(df["close"].iloc[-1]),
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


def fmt(sym: str, side: str, price: float, info: dict) -> str:
    arrow = "🟢 LONG" if side == "LONG" else "🔴 SHORT"
    return (
        f"{arrow}  <b>{sym}</b>  [HA-MOM 15m→6H]\n"
        f"{now_tr()}\n"
        f"Fiyat: <b>{price}</b>\n"
        f"6H HA EMA20 mom: {info['momentum_prev']} → {info['momentum']}\n"
        f"TV kontrol: 15m grafik + aynı strategy. Momentum 0 kesişimi.\n"
        f"<i>lookahead açık (kapanmamış 6H). Tavsiye değildir.</i>"
    )


def main() -> None:
    tickers = get_tickers()
    print(f"{len(tickers)} kripto taranacak")
    state = load_state()
    sent = 0
    for i, row in enumerate(tickers):
        sym = row.get("symbol")
        df = candles_6h(sym)
        time.sleep(0.04)
        if df is None:
            continue
        info = signal_from_6h(df)
        if not info or not info["side"]:
            continue
        side = info["side"]
        key = f"{sym}:{side}"
        if not cooldown_ok(state, key):
            print("cd", key)
            continue
        try:
            price = float(row.get("lastPr") or info["bar_close"])
        except (TypeError, ValueError):
            price = info["bar_close"]
        if send_tg(fmt(sym, side, price, info)):
            state.setdefault("last", {})[key] = datetime.now(timezone.utc).isoformat()
            sent += 1
            print("ok", key, price, info["momentum"])
        if sent >= 8:
            break
    save_state(state)
    print("bitti telegram", sent)


if __name__ == "__main__":
    main()
