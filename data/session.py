from dataclasses import dataclass

@dataclass(frozen=True)
class SessionConfig:
    tz: str = "Asia/Kolkata"; open: str = "09:15"; close: str = "15:30"; bar_minutes: int = 5
    @property
    def open_min(self): h, m = self.open.split(":"); return int(h) * 60 + int(m)
    @property
    def close_min(self): h, m = self.close.split(":"); return int(h) * 60 + int(m)
    @property
    def expected_bars(self): return (self.close_min - self.open_min) // self.bar_minutes

NSE = SessionConfig()
US = SessionConfig("America/New_York", "09:30", "16:00", 5)
