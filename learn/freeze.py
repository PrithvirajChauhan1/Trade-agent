import json, hashlib, datetime
from features.custom_indicators import (LearnedParams, AdaptiveEMAParams, RegimeRSIParams,
                                        VolBandParams, MomentumWeights)

def _p2d(p): return {"adaptive_ema": p.adaptive_ema._asdict(), "regime_rsi": p.regime_rsi._asdict(),
                     "vol_band": p.vol_band._asdict(), "momentum_weights": p.momentum_weights._asdict()}

def _d2p(d): return LearnedParams(AdaptiveEMAParams(**d["adaptive_ema"]), RegimeRSIParams(**d["regime_rsi"]),
                                  VolBandParams(**d["vol_band"]), MomentumWeights(**d["momentum_weights"]))

def freeze(path, params, combos, meta):
    payload = {"params": _p2d(params), "combos": combos, "meta": meta}
    body = json.dumps(payload, sort_keys=True, default=float)
    payload["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    payload["frozen_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    json.dump(payload, open(path, "w"), indent=2, default=float)
    return payload["sha256"]

def load_frozen(path):
    d = json.load(open(path)); sha = d.pop("sha256"); d.pop("frozen_at", None)
    if hashlib.sha256(json.dumps(d, sort_keys=True, default=float).encode()).hexdigest() != sha:
        raise ValueError("Frozen algorithm file was modified after freezing")
    return _d2p(d["params"]), d["combos"], d["meta"], sha
