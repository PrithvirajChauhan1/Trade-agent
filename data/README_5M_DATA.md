# Long-history 5m data

The real 5-minute research protocol requires an archived OHLCV dataset covering the
requested 2000-2025 period. The repository does not fabricate or bundle such data.

CSV requirements:
- datetime index in the first column
- `open,high,low,close,volume` columns
- timezone-aware timestamps preferred; otherwise timestamps are localized to the selected market session
- chronological, unique rows
- no missing OHLCV values

Run:

```bash
python -m research.run_research --source csv --csv /path/to/5m.csv \
  --train-start 2000-01-01 --train-end 2019-12-31 \
  --val-start 2018-01-01 --val-end 2019-12-31 \
  --test-start 2020-01-01 --test-end 2025-12-31 --cap 200
```

Yahoo Finance 5m is intentionally not used for this study because its historical
intraday retention is insufficient for the requested period.
