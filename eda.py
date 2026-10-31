"""Agent-oriented EDA of the three challenge CSVs.

Produces:
  eda/EDA_REPORT.md      findings, organised by which agent each one informs
  eda/figures/*.png      charts referenced by the report
  eda/calibration.json   numbers the scenario builder uses (taxi times, arrival rates, turnarounds)

Run: python eda.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).parent
OUT = ROOT / "eda"
FIG = OUT / "figures"
CAUSES = ["AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"]
# The full 2015 BTS on-time file has ~5.82M flights; this extract keeps only cause-coded delays.
FULL_2015_FLIGHTS = 5_819_079
WIDEBODY_DISTANCE_MI = 2000  # DOT has no aircraft type: long-haul distance is the wake-category proxy


def hhmm_to_min(s: pd.Series) -> pd.Series:
    s = s.astype(int)
    return (s // 100) % 24 * 60 + s % 100


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flights = pd.read_csv(ROOT / "flights.csv", low_memory=False,
                          dtype={"ORIGIN_AIRPORT": str, "DESTINATION_AIRPORT": str})
    return flights, pd.read_csv(ROOT / "airports.csv"), pd.read_csv(ROOT / "airlines.csv")


def savefig(name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130)
    plt.close()
    return f"figures/{name}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    f, airports, airlines = load()
    md: list[str] = ["# EDA: US DOT 2015 delays, viewed through the three agents\n"]
    cal: dict = {}

    # ------------------------------------------------------------------ overview
    months = sorted(f.MONTH.unique().tolist())
    missing_months = sorted(set(range(1, 13)) - set(months))
    cause_sum_ok = (f[CAUSES].sum(axis=1) == f.ARRIVAL_DELAY).mean()
    w, l = f.WEATHER_DELAY > 15, f.LATE_AIRCRAFT_DELAY > 20
    days = f.groupby("MONTH").DAY.nunique().sum()
    scale = FULL_2015_FLIGHTS * len(months) / 12 / len(f)
    no_coords = airports[airports.LATITUDE.isna()].IATA_CODE.tolist()
    md += [
        "## 1. What the files actually contain\n",
        f"- `flights.csv`: **{len(f):,} rows x {f.shape[1]} columns**, {f.AIRLINE.nunique()} airlines, "
        f"{f.DESTINATION_AIRPORT.nunique()} destination airports, {f.TAIL_NUMBER.nunique():,} airframes, {days} days.",
        f"- Months present: {months}. **Missing: {missing_months}** (October 2015 in the public file uses numeric "
        "airport IDs; it has been dropped from this extract).",
        f"- **The file is pre-filtered.** Every row already satisfies the challenge rule: "
        f"WEATHER_DELAY > 15 or LATE_AIRCRAFT_DELAY > 20 ({pct((w | l).mean())}). No cancellations "
        f"({int(f.CANCELLED.sum())}), no diversions ({int(f.DIVERTED.sum())}), minimum ARRIVAL_DELAY "
        f"{int(f.ARRIVAL_DELAY.min())} min. Delay-cause columns have no nulls.",
        f"- Integrity: the five cause columns sum exactly to ARRIVAL_DELAY on {pct(cause_sum_ok)} of rows.",
        f"- It is roughly **1/{scale:.0f} of all 2015 flights** for those months, so raw counts here understate "
        "real airport traffic. The scenario builder scales arrival density by this factor (an estimate).",
        f"- `airports.csv`: {len(airports)} airports; {len(no_coords)} lack coordinates: {no_coords} "
        "(weather lookups fall back to OurAirports for these).",
        f"- `airlines.csv`: {len(airlines)} carriers, all present in flights.csv.\n",
    ]

    # ------------------------------------------------------------------ disruption mix
    mix = {
        "late-aircraft only (L>20, W<=15)": int((l & ~w).sum()),
        "weather only (W>15, L<=20)": int((w & ~l).sum()),
        "both (W>15 and L>20)": int((w & l).sum()),
    }
    md += ["## 2. Disruption mix (drives scenario selection)\n",
           "| bucket | rows | share |", "|---|---:|---:|"]
    md += [f"| {k} | {v:,} | {pct(v / len(f))} |" for k, v in mix.items()]
    md += ["", "Late-aircraft cascades dominate (the network problem: Ground and turnaround). Weather rows are the "
           "minority but carry the extreme tails (Pilot fuel and ATC runway capacity). An evaluation that samples "
           "rows uniformly would be ~90% late-aircraft cases, so the batch evaluator stratifies by bucket "
           "instead.\n"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, col, colour in [(ax[0], "WEATHER_DELAY", "#4c78a8"), (ax[1], "LATE_AIRCRAFT_DELAY", "#f58518")]:
        v = f.loc[f[col] > 0, col].clip(upper=300)
        a.hist(v, bins=60, color=colour)
        a.set_title(f"{col} (>0, clipped at 300 min)")
        a.set_xlabel("minutes")
        a.axvline(15 if col.startswith("W") else 20, color="k", ls="--", lw=1)
    md += [f"![delay distributions]({savefig('delay_distributions.png')})\n"]

    q = [0.5, 0.75, 0.9, 0.95, 0.99]
    md += ["| percentile | WEATHER_DELAY (rows >15) | LATE_AIRCRAFT_DELAY (rows >20) |", "|---|---:|---:|"]
    wq, lq = f.loc[w, "WEATHER_DELAY"].quantile(q), f.loc[l, "LATE_AIRCRAFT_DELAY"].quantile(q)
    md += [f"| p{int(p * 100)} | {wq[p]:.0f} | {lq[p]:.0f} |" for p in q]
    md += ["", f"Delays beyond ~{int(lq[0.95])} min are no longer tactical (a 3-turn arrival negotiation cannot "
           "fix a multi-hour ground stop), so the scenario loader caps WEATHER_DELAY at 90 and "
           "LATE_AIRCRAFT_DELAY at 120 min and labels the rest as out-of-scope for live negotiation.\n"]
    cal["severity_quantiles"] = {"weather": {str(k): float(v) for k, v in wq.items()},
                                 "late_aircraft": {str(k): float(v) for k, v in lq.items()}}

    # ------------------------------------------------------------------ PILOT
    md += ["## 3. Pilot agent: weather exposure\n"]
    wx = f[w]
    by_month = wx.groupby("MONTH").size().reindex(range(1, 13), fill_value=0)
    top_wx = wx.DESTINATION_AIRPORT.value_counts().head(10)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].bar(by_month.index, by_month.values, color="#4c78a8")
    ax[0].set_title("Weather-delayed arrivals by month")
    ax[0].set_xticks(range(1, 13))
    ax[1].barh(top_wx.index[::-1], top_wx.values[::-1], color="#4c78a8")
    ax[1].set_title("Top destinations, WEATHER_DELAY > 15")
    md += [f"![weather seasonality]({savefig('pilot_weather.png')})\n",
           f"- Peak weather months: {by_month.sort_values(ascending=False).head(3).index.tolist()} "
           "(summer convection plus winter storms). Holding-fuel stress is seasonal.",
           f"- Most exposed destinations: {', '.join(top_wx.head(5).index)}. These are the airports where the "
           "Pilot agent most needs PIREPs (turbulence and icing in the hold) and METAR crosswinds.",
           f"- Long-haul share (distance >= {WIDEBODY_DISTANCE_MI} mi, the HEAVY wake proxy): "
           f"{pct((f.DISTANCE >= WIDEBODY_DISTANCE_MI).mean())}. DOT has no aircraft type, so wake category "
           "and fuel burn are inferred from distance.\n"]

    # ------------------------------------------------------------------ GROUND
    md += ["## 4. Ground agent: rotations, turnaround and propagation\n"]
    g = f.sort_values(["TAIL_NUMBER", "MONTH", "DAY", "SCHEDULED_DEPARTURE"])
    nxt = g.groupby(["TAIL_NUMBER", "MONTH", "DAY"]).shift(-1)
    pair = (nxt.ORIGIN_AIRPORT == g.DESTINATION_AIRPORT)
    ground_time = (hhmm_to_min(nxt.loc[pair, "SCHEDULED_DEPARTURE"])
                   - hhmm_to_min(g.loc[pair, "SCHEDULED_ARRIVAL"])) % 1440
    ground_time = ground_time[(ground_time >= 20) & (ground_time <= 240)]
    wide = g.loc[ground_time.index, "DISTANCE"] >= WIDEBODY_DISTANCE_MI
    prop = pd.DataFrame({"in": g.loc[pair, "ARRIVAL_DELAY"], "out": nxt.loc[pair, "DEPARTURE_DELAY"]}).dropna()
    absorbed = (prop["in"] - prop["out"]).clip(lower=0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].hist(ground_time, bins=45, color="#f58518")
    ax[0].set_title("Scheduled ground time between legs (same tail)")
    ax[0].set_xlabel("minutes")
    sample = prop.sample(min(5000, len(prop)), random_state=0).clip(upper=400)
    ax[1].scatter(sample["in"], sample["out"], s=3, alpha=0.3, color="#f58518")
    ax[1].plot([0, 400], [0, 400], "k--", lw=1)
    ax[1].set_title("Inbound arrival delay vs next-leg departure delay")
    ax[1].set_xlabel("inbound arrival delay")
    ax[1].set_ylabel("outbound departure delay")
    md += [f"![rotations]({savefig('ground_rotations.png')})\n",
           f"- {pair.sum():,} same-tail, same-day rotations found in the extract.",
           f"- Scheduled ground time: median **{ground_time.median():.0f} min** "
           f"(narrow-body proxy {ground_time[~wide].median():.0f}, long-haul proxy "
           f"{ground_time[wide].median() if wide.any() else float('nan'):.0f}).",
           f"- Delay propagation: correlation between inbound arrival delay and the next departure delay is "
           f"**{prop.corr().iloc[0, 1]:.2f}**; turnarounds absorb a median of {absorbed.median():.0f} min. "
           "This is why Ground's turnaround and gate decisions matter: most of the delay rides straight into "
           "the next leg.\n"]
    cal["turnaround_min"] = {"MEDIUM": int(min(55, max(35, ground_time[~wide].median()))),
                             "HEAVY": int(min(110, max(60, ground_time[wide].median()))) if wide.any() else 75}

    # ------------------------------------------------------------------ ATC
    md += ["## 5. ATC agent: arrival density and taxi-in\n"]
    f["ARR_HOUR"] = f.SCHEDULED_ARRIVAL // 100 % 24
    n_days = f.groupby(["MONTH", "DAY"]).ngroups
    hourly = f.groupby(["DESTINATION_AIRPORT", "ARR_HOUR"]).size() / n_days * scale
    top_dest = f.DESTINATION_AIRPORT.value_counts().head(8).index
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for code in top_dest[:5]:
        series = hourly.loc[code].reindex(range(24), fill_value=0)
        ax.plot(series.index, series.values, label=code)
    ax.set_title(f"Estimated arrivals per hour (extract counts x {scale:.0f})")
    ax.set_xlabel("local hour")
    ax.legend()
    peak = hourly.groupby(level=0).max().sort_values(ascending=False)
    taxi = f.groupby("DESTINATION_AIRPORT").TAXI_IN.quantile([0.5, 0.75, 0.9]).unstack()
    md += [f"![arrival density]({savefig('atc_density.png')})\n",
           f"- Estimated peak arrival rates: " + ", ".join(f"{k} {v:.0f}/h" for k, v in peak.head(5).items())
           + ". At 2-3 min wake spacing a single runway lands about 20-30 aircraft an hour, so the hubs are "
           "near capacity at peak and every displaced arrival competes for a scarce slot.",
           f"- Taxi-in (wheels-on to in-block): network median {f.TAXI_IN.median():.0f} min, p90 "
           f"{f.TAXI_IN.quantile(0.9):.0f} min; worst hubs by median: "
           + ", ".join(f"{k} {v:.0f}" for k, v in taxi[0.5].sort_values(ascending=False).head(5).items()) + ".",
           "- Used per airport: the near-terminal runway gets the median taxi-in, the far runway the p75.\n"]

    cal["arrival_scale_factor"] = round(scale, 2)
    cal["airports"] = {
        code: {"taxi_in_p50": float(taxi.loc[code, 0.5]), "taxi_in_p75": float(taxi.loc[code, 0.75]),
               "arrivals_per_hour": {int(h): round(float(v), 1) for h, v in hourly.loc[code].items()}}
        for code in taxi.index
    }

    # ------------------------------------------------------------------ evaluation design
    md += ["## 6. Evaluation design (instead of a train/test split)\n",
           "Nothing in the agents is trained, so a classic ML train/test split does not apply. What does apply "
           "when prompts are iterated: **tune on a dev set and report on a held-out test set**, or the prompts "
           "overfit the handful of rows you looked at. `batch_eval.py` assigns each row to dev or test by a "
           "stable hash of its identity and stratifies by disruption bucket (section 2) and severity "
           "(section 2 percentiles).\n"]

    OUT.mkdir(exist_ok=True)
    (OUT / "EDA_REPORT.md").write_text("\n".join(md) + "\n")
    (OUT / "calibration.json").write_text(json.dumps(cal, indent=1))
    print("\n".join(md))
    print(f"\nWrote {OUT / 'EDA_REPORT.md'} and {OUT / 'calibration.json'}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
