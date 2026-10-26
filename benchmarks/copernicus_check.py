"""
A `demo.py` run's monthly means against Copernicus's: Climate Pulse's daily
global mean 2 m temperature from ERA5, each day the mean of its hourly values,
averaged over the days of each month in the run's window. It charts both,
and their difference below, beside the run, as `results/demo-....-copernicus.svg`:

    uv run python benchmarks/copernicus_check.py results/demo-....json
"""

import argparse
import csv
import io
import json
import urllib.request
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

DAILY = "https://sites.ecmwf.int/data/climatepulse/data/series/era5_daily_series_2t_global.csv"

# Climate Pulse rounds each day to 0.001 °C
TOLERANCE_C = 0.001


def daily_means() -> dict[date, float]:
    with urllib.request.urlopen(DAILY) as response:
        text = response.read().decode()
    rows = csv.DictReader(line for line in io.StringIO(text) if not line.startswith("#"))
    return {date.fromisoformat(row["date"]): float(row["2t"]) for row in rows}


def chart(months: list[datetime], ours: list[float], copernicus: list[float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    marker = "." if len(months) < 50 else None
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    # Copernicus wide and pale beneath, so where the two agree it shows as an outline
    top.plot(months, copernicus, color="tab:orange", linewidth=4, alpha=0.5, label="Copernicus")
    top.plot(months, ours, color="tab:blue", linewidth=1, marker=marker, label="map-lambda")
    top.set_ylabel("°C")
    top.set_title(f"Monthly global mean 2 m temperature, {len(months)} months")
    top.legend(loc="upper right", frameon=False)

    differences = [a - e for a, e in zip(ours, copernicus, strict=True)]
    bottom.axhspan(
        -TOLERANCE_C / 2,
        TOLERANCE_C / 2,
        color="tab:orange",
        alpha=0.2,
        linewidth=0,
        label="Copernicus's rounding",
    )
    bottom.axhline(0, color="0.6", linewidth=0.8)
    bottom.plot(months, differences, color="tab:blue", linewidth=0.8, marker=marker)
    bottom.set_ylabel("difference (°C)")
    bottom.legend(loc="upper right", frameon=False)
    for ax in (top, bottom):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    print(f"chart in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="a saved demo.py run")
    args = parser.parse_args()
    run = json.loads(args.results.read_text())
    start, end = (datetime.fromisoformat(run[k]) for k in ("start", "end"))
    if any(t != datetime.combine(t.date(), datetime.min.time()) for t in (start, end)):
        raise SystemExit("Climate Pulse is daily, so the run's window must be whole days")

    by_month = defaultdict(list)
    for day, mean in daily_means().items():
        if start.date() <= day < end.date():
            by_month[f"{day:%Y-%m}"].append(mean)

    months: list[datetime] = []
    ours: list[float] = []
    copernicus: list[float] = []
    for row in run["rows"]:
        month = row["month"][:7]
        days = by_month[month]
        if not days:
            raise SystemExit(f"Climate Pulse has no days for {month}")
        expected = sum(days) / len(days)
        actual = row["mean_t2m"] - 273.15
        months.append(datetime.fromisoformat(row["month"]))
        ours.append(actual)
        copernicus.append(expected)
        print(f"{month}  Copernicus {expected:7.3f}  map-lambda {actual:8.4f} °C")
    chart(months, ours, copernicus, args.results.with_name(f"{args.results.stem}-copernicus.svg"))
    differences = [a - e for a, e in zip(ours, copernicus, strict=True)]
    print(f"{len(months)} months, max difference {max(map(abs, differences)):.4f} °C")
    over = [(m, d) for m, d in zip(months, differences, strict=True) if abs(d) >= TOLERANCE_C]
    if not over:
        print("all as expected")
        return
    print(f"over the {TOLERANCE_C} °C tolerance:")
    for month, difference in over:
        print(f"  {month:%Y-%m}  {difference:+.4f} °C")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
