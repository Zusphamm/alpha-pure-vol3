"""
Pure Vol 3 — standalone backtest with IS/OS protocol and cost.
Factors: parkinson_vol30|chg20|- + ivol|z20|- + residual_vol|z20|-
Liq filter: Volume rank top 100 (no QV dependency)
"""
import numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path.home() / "Desktop/datacryp/derived/panel_wide/1d"
COST_ONEWAY = 0.0007
ALPHAS = ["parkinson_vol30|chg20|-", "ivol|z20|-", "residual_vol|z20|-"]

def load_panels():
    panels = {}
    for f in ["close", "high", "low", "volume", "open"]:
        p = DATA_DIR / f"{f}.parquet"
        if p.exists(): panels[f] = pd.read_parquet(p)
    return panels

def compute_factor(name, panels, liq):
    close, high, low = panels["close"], panels["high"], panels["low"]
    ret = close.pct_change()
    parts = name.split("|")
    fn, tf, dr = parts[0], parts[1], parts[2]
    sgn = 1 if dr == "+" else -1
    if fn == "parkinson_vol30":
        loghl = np.log(high / low)
        raw = (loghl**2 / (4*np.log(2))).rolling(30, 15).mean().apply(np.sqrt) * np.sqrt(365)
    elif fn == "ivol":
        raw = ret.sub(ret.mean(axis=1), axis=0).rolling(20, 10).std() * np.sqrt(365)
    elif fn == "residual_vol":
        raw = ret.sub(ret.mean(axis=1), axis=0).rolling(20, 10).std()
    else: return None
    if tf == "lvl": F = raw
    elif tf == "chg20": F = raw - raw.shift(20)
    elif tf == "z20": F = (raw - raw.rolling(20,10).mean()) / raw.rolling(20,10).std().replace(0,np.nan)
    else: F = raw
    sig = F.where(liq)
    sig = sig.sub(sig.mean(axis=1), axis=0).div(sig.std(axis=1).replace(0,np.nan), axis=0) * sgn
    m, s = sig.mean(axis=1), sig.std(axis=1)
    sig = sig.clip(lower=(m-3*s), upper=(m+3*s), axis=0)
    g = sig.abs().sum(axis=1).replace(0,np.nan)
    return sig.div(g, axis=0).fillna(0)

def backtest(start, end=None, top_n=30):
    panels = load_panels()
    ret = panels["close"].pct_change()
    liq = panels["volume"].rank(axis=1, ascending=False) <= 100
    ens = None; wt = 1.0/len(ALPHAS)
    for a in ALPHAS:
        w = compute_factor(a, panels, liq)
        if w is None: continue
        ens = w*wt if ens is None else ens.add(w*wt, fill_value=0)
    if ens is None: return None
    mask = ens.abs().rank(axis=1, ascending=False) <= top_n
    pos = ens.where(mask, 0)
    d = np.sign(pos); n = (d!=0).sum(axis=1).replace(0,np.nan)
    pos = d.div(n, axis=0).fillna(0).shift(1)
    gross = (pos * ret).sum(axis=1)
    turn = pos.diff().abs().sum(axis=1)
    net = gross - turn * COST_ONEWAY
    pnl = net.loc[start:end].dropna() if end else net.loc[start:].dropna()
    if len(pnl) < 30: return None
    sr = pnl.mean()/pnl.std()*np.sqrt(365) if pnl.std()>0 else 0
    cum = pnl.cumsum(); dd = (cum-cum.cummax()).min()
    wr = (pnl>0).mean()*100
    return {'sr':round(sr,2),'ret':round(cum.iloc[-1]*100,1),'dd':round(dd*100,1),'wr':round(wr,1),'days':len(pnl),'pnl':pnl}

if __name__ == "__main__":
    print(f"Pure Vol 3 Backtest (cost={COST_ONEWAY*100:.2f}% one-way)")
    print(f"Factors: {' + '.join(ALPHAS)}")
    print(f"Liq: Volume rank top 100 (no QV)")
    print()
    for pname, s, e in [("Full 2019+","2019-01-01",None),("IS 2021-23","2021-01-01","2023-12-31"),
                         ("OS 2024+","2024-01-01",None),("2026","2026-01-01",None)]:
        r = backtest(s, e)
        if r: print(f"  {pname:<14s} Sharpe={r['sr']:>5.2f} Ret={r['ret']:>7.1f}% DD={r['dd']:>6.1f}% WR={r['wr']:.1f}% Days={r['days']}")

    print("\nYearly Sharpe:")
    r = backtest("2019-01-01")
    if r:
        for yr in sorted(r['pnl'].index.year.unique()):
            y = r['pnl'][r['pnl'].index.year==yr]
            if len(y)>20:
                ysr = y.mean()/y.std()*np.sqrt(365) if y.std()>0 else 0
                print(f"  {yr}: {ysr:>6.2f}{'  << NEGATIVE' if ysr<0 else ''}")

    is_r = backtest("2021-01-01","2023-12-31"); os_r = backtest("2024-01-01")
    if is_r and os_r:
        decay = os_r['sr']/is_r['sr'] if is_r['sr']!=0 else 0
        print(f"\nIS/OS Protocol: IS={is_r['sr']} OS={os_r['sr']} Decay={decay:.2f}")
        print(f"  IS>=1.4: {'PASS' if is_r['sr']>=1.4 else 'FAIL'} | OS>0: {'PASS' if os_r['sr']>0 else 'FAIL'} | DD>-50: {'PASS' if os_r['dd']>-50 else 'FAIL'} | Decay>=0.3: {'PASS' if decay>=0.3 else 'FAIL'}")
        print(f"  Result: {'>>> PASS <<<' if is_r['sr']>=1.4 and os_r['sr']>0 and os_r['dd']>-50 and decay>=0.3 else 'FAIL'}")
