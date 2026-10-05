"""
Pure Vol 3 — Cross-sectional alpha for crypto perpetual futures.
Factors: parkinson_vol30|chg20|- + ivol|z20|- + residual_vol|z20|-

Backtest (clean 1d data, 7bp cost):
  IS 2021-2023: Sharpe 1.44 | OS 2024+: Sharpe 1.94 | Decay 1.35
  2026 YTD: Sharpe 4.11 | Max DD -46%

No QV dependency — uses Volume rank for universe, pure OHLC factors.
"""
import sys, os, json, time, logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [pure-vol3] %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "trader.log"), logging.StreamHandler()],
)
log = logging.getLogger("pure-vol3")

DATA_DIR = Path.home() / "Desktop/datacryp/derived/panel_wide"
STATE_FILE = Path(__file__).parent / "live_state.json"
REBALANCE_SNAP_FILE = Path(__file__).parent / "rebalance_snap.json"

ALPHAS = [
    {"name": "parkinson_vol30|chg20|-", "weight": 1/3},
    {"name": "ivol|z20|-", "weight": 1/3},
    {"name": "residual_vol|z20|-", "weight": 1/3},
]
TOP_N = 30
LEVERAGE = 5
GROSS_TARGET = 5.5
FREE_PCT = 0.33
MIN_NOTIONAL = 5.0
REBALANCE_INTERVAL = 3600
DRY_RUN = os.getenv("LIVE_DRY_RUN", "1") == "1"
EQUAL_WEIGHT = True
FUNDING_FILTER = True
FUNDING_FILTER_THRESHOLD = 0.03


def load_panels(tf="1d"):
    panels = {}
    for field in ["close", "volume", "open", "high", "low"]:
        p = DATA_DIR / tf / f"{field}.parquet"
        if p.exists():
            panels[field] = pd.read_parquet(p)
    return panels if panels else None


def _compute_factor(name, panels, liq):
    close, high, low = panels["close"], panels["high"], panels["low"]
    volume = panels["volume"]
    ret = close.pct_change()

    parts = name.split("|")
    factor_name, transform, direction = parts[0], parts[1], parts[2]
    sgn = 1 if direction == "+" else -1

    if factor_name == "parkinson_vol30":
        loghl = np.log(high / low)
        raw = (loghl ** 2 / (4 * np.log(2))).rolling(30, min_periods=15).mean().apply(np.sqrt) * np.sqrt(365)
    elif factor_name == "ivol":
        mkt = ret.mean(axis=1)
        raw = ret.sub(mkt, axis=0).rolling(20, min_periods=10).std() * np.sqrt(365)
    elif factor_name == "residual_vol":
        mkt = ret.mean(axis=1)
        raw = ret.sub(mkt, axis=0).rolling(20, min_periods=10).std()
    else:
        log.warning(f"Unknown factor: {factor_name}")
        return None

    if transform == "lvl": F = raw
    elif transform == "chg20": F = raw - raw.shift(20)
    elif transform == "chg60": F = raw - raw.shift(60)
    elif transform == "z20": F = (raw - raw.rolling(20, 10).mean()) / raw.rolling(20, 10).std().replace(0, np.nan)
    elif transform == "z60": F = (raw - raw.rolling(60, 30).mean()) / raw.rolling(60, 30).std().replace(0, np.nan)
    elif transform == "z120": F = (raw - raw.rolling(120, 60).mean()) / raw.rolling(120, 60).std().replace(0, np.nan)
    else: F = raw

    sig = F.where(liq)
    sig = sig.sub(sig.mean(axis=1), axis=0).div(sig.std(axis=1).replace(0, np.nan), axis=0) * sgn
    m, s = sig.mean(axis=1), sig.std(axis=1)
    sig = sig.clip(lower=(m - 3 * s), upper=(m + 3 * s), axis=0)
    g = sig.abs().sum(axis=1).replace(0, np.nan)
    return sig.div(g, axis=0).fillna(0)


def _fetch_funding_rates():
    try:
        ex = ccxt.binance({"options": {"defaultType": "future"}, "enableRateLimit": True})
        funding = ex.fetch_funding_rates()
        return {sym.split("/")[0].split(":")[0] + "USDT": float(info.get("fundingRate", 0)) * 100
                for sym, info in funding.items()}
    except Exception as e:
        log.warning(f"Funding fetch failed: {e}")
        return {}


def compute_signals():
    panels = load_panels("1d")
    if not panels:
        log.error("No panel data")
        return {}, None

    liq = panels["volume"].rank(axis=1, ascending=False) <= 100

    ensemble = None
    for a in ALPHAS:
        w = _compute_factor(a["name"], panels, liq)
        if w is None: continue
        weighted = w * a["weight"]
        ensemble = weighted if ensemble is None else ensemble.add(weighted, fill_value=0)

    if ensemble is None:
        log.error("No signals")
        return {}, None

    last = ensemble.iloc[-1].dropna()
    top = last.abs().sort_values(ascending=False).head(TOP_N)
    last = last.reindex(top.index)

    if EQUAL_WEIGHT:
        dirs = np.sign(last)
        n = len(last)
        if n > 0:
            last = dirs * (GROSS_TARGET / n)

    data_date = str(panels["close"].index[-1])
    data_dt = pd.Timestamp(data_date)
    now_utc = pd.Timestamp.now(tz="UTC")
    if (data_dt.tz_localize("UTC") if data_dt.tz is None else data_dt) >= now_utc.normalize():
        log.info(f"  Candle {data_date} not closed yet, skip")
        return {}, data_date

    if FUNDING_FILTER:
        rates = _fetch_funding_rates()
        for sym in list(last.index):
            rate = rates.get(sym, 0)
            wt = last[sym]
            if (wt > 0 and rate > FUNDING_FILTER_THRESHOLD) or (wt < 0 and rate < -FUNDING_FILTER_THRESHOLD):
                log.info(f"  Skip {sym}: funding={rate:+.4f}%")
                last = last.drop(sym)

    positions = {sym: round(float(wt), 8) for sym, wt in last.items() if abs(wt) > 1e-6}
    n_l = sum(1 for v in positions.values() if v > 0)
    n_s = sum(1 for v in positions.values() if v < 0)
    gross = sum(abs(v) for v in positions.values())
    log.info(f"  PURE_VOL3: {len(positions)} pos (L={n_l} S={n_s}), gross={gross:.3f}")
    return positions, data_date


def init_exchange():
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        log.error("BINANCE_API_KEY / BINANCE_API_SECRET not set")
        return None
    ex = ccxt.binance({
        "apiKey": api_key, "secret": api_secret,
        "options": {"defaultType": "future"}, "enableRateLimit": True,
    })
    ex.set_sandbox_mode(False)
    return ex


def load_market_info(ex):
    markets = ex.load_markets()
    info = {}
    for sym, m in markets.items():
        if (m.get("future") or m.get("swap")) and m.get("active") and ":USDT" in sym and "USDT-" not in sym:
            info[sym] = {
                "min_qty": float(m["limits"]["amount"]["min"] or 0),
                "qty_step": float(m["precision"]["amount"] or 0.001),
                "min_notional": float(m["limits"]["cost"]["min"] or MIN_NOTIONAL),
            }
    return info


def resolve_symbol(sym, market_info):
    base = sym.replace("USDT", "") if sym.endswith("USDT") else sym
    for c in [f"{base}/USDT:USDT", f"{sym}/USDT:USDT", f"{sym}:USDT"]:
        if c in market_info:
            return c
    return None


def round_qty(qty, step):
    if step <= 0: return qty
    precision = max(0, -int(np.floor(np.log10(step))))
    return round(np.floor(qty / step) * step, precision)


def rebalance(ex, target_weights, capital):
    market_info = load_market_info(ex)
    positions = ex.fetch_positions()
    current = {}
    for p in positions:
        if abs(float(p["contracts"])) > 0:
            amt = float(p["contracts"]) * (1 if p["side"] == "long" else -1)
            current[p["symbol"]] = {"amount": amt, "notional": abs(float(p.get("notionalValue") or p.get("notional") or 0))}

    sym_map = {}
    for sym in target_weights:
        resolved = resolve_symbol(sym, market_info)
        if resolved: sym_map[sym] = resolved

    all_symbols = set(sym_map.values()) | set(current.keys())
    tickers = ex.fetch_tickers(list(all_symbols)) if all_symbols else {}
    prices = {s: float(tickers[s]["last"]) for s in all_symbols if s in tickers}

    merged_gross = sum(abs(v) for v in target_weights.values()) or 1

    orders = []
    for sym, weight in target_weights.items():
        ccxt_sym = sym_map.get(sym)
        if not ccxt_sym or ccxt_sym not in prices: continue
        minfo = market_info.get(ccxt_sym, {})
        price = prices[ccxt_sym]
        target_notional = abs(weight) / merged_gross * capital
        if target_notional < max(minfo.get("min_notional", MIN_NOTIONAL), MIN_NOTIONAL): continue
        target_qty = target_notional / price
        target_qty = round_qty(target_qty, minfo.get("qty_step", 0.001))
        target_amt = target_qty if weight > 0 else -target_qty
        cur_amt = current.get(ccxt_sym, {}).get("amount", 0)
        diff = target_amt - cur_amt
        if abs(diff) * price < MIN_NOTIONAL: continue
        orders.append({"symbol": ccxt_sym, "side": "buy" if diff > 0 else "sell",
                       "qty": round_qty(abs(diff), minfo.get("qty_step", 0.001)),
                       "notional": abs(diff) * price})

    for sym in current:
        if sym not in sym_map.values():
            cur = current[sym]
            if abs(cur["amount"]) > 0:
                orders.append({"symbol": sym, "side": "sell" if cur["amount"] > 0 else "buy",
                               "qty": abs(cur["amount"]), "notional": cur["notional"]})

    log.info(f"  {len(orders)} orders")
    if DRY_RUN:
        for o in orders:
            log.info(f"    [DRY] {o['side']} {o['qty']} {o['symbol']} ${o['notional']:.0f}")
        return

    executed = 0
    for o in orders:
        try:
            try: ex.set_leverage(LEVERAGE, o["symbol"])
            except: pass
            ex.create_order(symbol=o["symbol"], type="market", side=o["side"], amount=o["qty"])
            log.info(f"    OK: {o['side']} {o['qty']} {o['symbol']}")
            executed += 1
            time.sleep(0.1)
        except Exception as e:
            log.error(f"    FAIL: {o['symbol']} — {e}")
    log.info(f"  Executed: {executed}/{len(orders)}")


def main():
    log.info("=" * 60)
    log.info("PURE VOL 3 — Live Trader (Binance Futures)")
    log.info(f"Factors: {[a['name'] for a in ALPHAS]}")
    log.info(f"Leverage: {LEVERAGE}x | Gross: {GROSS_TARGET} | Free: {FREE_PCT*100:.0f}% | DRY: {DRY_RUN}")
    log.info("=" * 60)

    ex = init_exchange()
    if not ex: return
    last_data_date = None

    while True:
        try:
            target, data_date = compute_signals()
            if not target:
                time.sleep(300)
                continue
            if data_date == last_data_date:
                log.info(f"  Same date {data_date}, skip")
                time.sleep(REBALANCE_INTERVAL)
                continue

            bal = ex.fetch_balance()
            wallet = float(bal["total"].get("USDT", 0))
            target_margin = wallet * (1 - FREE_PCT)
            target_gross = target_margin * LEVERAGE
            merged_gross = sum(abs(v) for v in target.values()) or 1
            capital = target_gross / merged_gross
            free_pct = 1 - (target_margin / wallet) if wallet > 0 else 1
            log.info(f"  Wallet: ${wallet:.2f} | Capital: ${capital:.0f} | Free: {free_pct*100:.0f}%")

            rebalance(ex, target, capital)
            last_data_date = data_date

            REBALANCE_SNAP_FILE.write_text(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "wallet": round(wallet, 2),
            }, indent=2))
            STATE_FILE.write_text(json.dumps({
                "last_rebalance": datetime.now(timezone.utc).isoformat(),
                "data_date": data_date, "capital": round(capital, 2),
                "equity": round(wallet, 2), "n_positions": len(target),
            }, indent=2))

        except Exception as e:
            log.error(f"Error: {e}")

        time.sleep(REBALANCE_INTERVAL)


if __name__ == "__main__":
    main()
