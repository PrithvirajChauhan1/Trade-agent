import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

def _candles(ax, d):
    x = np.arange(len(d)); up = d.close >= d.open
    ax.vlines(x, d.low, d.high, color="k", lw=0.6)
    ax.bar(x, (d.close - d.open).abs().replace(0, 1e-9), bottom=np.minimum(d.open, d.close), color=np.where(up, "#2a9d8f", "#e76f51"), width=0.7)
    return x

def session_panel(df, session_date, path):
    d = df[df.session_date == session_date]; x = np.arange(len(d))
    fig, ax = plt.subplots(7, 1, figsize=(11, 14), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1, 1, 1.2, 1, 1]})
    _candles(ax[0], d); [ax[0].plot(x, d[c], lw=1, label=c) for c in ("ema_9", "ema_21", "ema_50")]; ax[0].legend(fontsize=7); ax[0].set_title(f"{session_date}: price + EMA")
    ax[1].plot(x, d.close, "k", lw=.8); ax[1].plot(x, d.vwap, "b", lw=1, label="session VWAP"); ax[1].legend(fontsize=7)
    ax[2].plot(x, d.rsi_14); ax[2].axhline(70, c="r", lw=.5); ax[2].axhline(30, c="g", lw=.5); ax[2].set_ylabel("RSI")
    ax[3].bar(x, d["hist"], color="gray"); ax[3].plot(x, d.macd, "b", lw=.8); ax[3].plot(x, d.signal, "r", lw=.8); ax[3].set_ylabel("MACD")
    ax[4].bar(x, d.volume, color="lightsteelblue"); a2 = ax[4].twinx(); a2.plot(x, d.rel_volume_tod, "r", lw=.8); ax[4].set_ylabel("Volume / rel")
    ax[5].plot(x, d.adx_14); ax[5].axhline(25, c="gray", lw=.5); ax[5].set_ylabel("ADX")
    ax[6].plot(x, d.close, "k", lw=.8); [ax[6].plot(x, d[c], "c", lw=.8) for c in ("bb_upper", "bb_mid", "bb_lower")]; ax[6].set_ylabel("BB")
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)

def trade_chart(df, t, path, pad_before=30, pad_after=8):
    ia = df.index.get_loc(t.entry_time); ib = df.index.get_loc(t.exit_time)
    lo, hi = max(0, ia - pad_before), min(len(df), ib + pad_after); d = df.iloc[lo:hi]
    fig, ax = plt.subplots(figsize=(10, 5)); _candles(ax, d); ex, xx = ia - lo, ib - lo
    ax.plot(np.arange(len(d)), d.vwap, "b", lw=.8, alpha=.6)
    ax.axhline(t.stop, c="r", ls="--", lw=1); ax.axhline(t.target, c="g", ls="--", lw=1)
    ax.scatter([ex], [t.entry], marker="^" if t.direction == "long" else "v", s=120, c="blue", zorder=5, label="ENTRY")
    ax.scatter([xx], [t.exit], marker="X", s=120, c="k", zorder=5, label=f"EXIT ({t.exit_reason})")
    ax.text(len(d) - 1, t.stop, " STOP", c="r", va="center"); ax.text(len(d) - 1, t.target, " TARGET", c="g", va="center")
    ax.set_title(f"{t.strategy} {t.direction} | net {t.net_r:+.2f}R | {t.entry_time:%Y-%m-%d %H:%M}"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)
