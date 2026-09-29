"""Versioned strategy registry. Configs are immutable per (id, version); status changes are validated;
PAPER_TRADING (and beyond) can never be set without an explicit human approver."""
import json, hashlib, os, datetime as dt
PATH = "strategy_store/registry.json"
ALLOWED = {"EXPERIMENTAL": {"VALIDATED", "REJECTED", "DEGRADED"}, "VALIDATED": {"DEGRADED", "REJECTED", "PAPER_TRADING"},
           "DEGRADED": {"EXPERIMENTAL", "REJECTED", "RETIRED"}, "PAPER_TRADING": {"DEGRADED", "RETIRED"},
           "REJECTED": {"RETIRED"}, "RETIRED": set()}
NEEDS_APPROVAL = {"PAPER_TRADING"}

def _h(cfg): return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]

class Registry:
    def __init__(self, path=PATH):
        self.path = path; self.data = json.load(open(path)) if os.path.exists(path) else {}
    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True); json.dump(self.data, open(self.path, "w"), indent=2, default=str)
    @staticmethod
    def key(cfg): return f"{cfg['id']}_{int(cfg['version']):03d}"
    def register(self, cfg, training_period, features):
        k = self.key(cfg); core = {x: cfg[x] for x in cfg if x != "hypothesis"}
        if k in self.data:
            if self.data[k]["config_hash"] != _h(core): raise ValueError(f"{k}: config changed — create a NEW version, never overwrite")
            return k
        self.data[k] = {"strategy_id": cfg["id"], "version": cfg["version"], "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                        "training_period": training_period, "parameters": core, "config_hash": _h(core), "hypothesis": cfg.get("hypothesis", ""),
                        "features": features, "status": "EXPERIMENTAL", "status_history": [], "runs": []}
        self._save(); return k
    def record_run(self, k, run): self.data[k]["runs"].append(run); self._save()
    def set_status(self, k, status, reason, approved_by=None):
        cur = self.data[k]["status"]
        if cur == status: return
        if status not in ALLOWED[cur]: raise ValueError(f"{k}: illegal transition {cur} -> {status}")
        if status in NEEDS_APPROVAL and not approved_by: raise PermissionError(f"{status} requires explicit human approval (approved_by)")
        self.data[k]["status_history"].append({"from": cur, "to": status, "reason": reason, "approved_by": approved_by, "at": dt.datetime.now().isoformat(timespec="seconds")})
        self.data[k]["status"] = status; self._save()
