"""
Strava Weekly Heart Rate Trend
---------------------------------
Fetches ~3 years of running data from Strava and plots weekly avg and
median heart rate over time.

A downward trend = lower HR over time = improved aerobic fitness.

Requirements:
    pip install requests matplotlib numpy python-dotenv

Usage:
    python strava_monthly_comparison.py
"""

import os
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict
from datetime import datetime, timedelta, timezone

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ── Config ───────────────────────────────────────────────────────────────────
CLIENT_ID = os.environ["STRAVA_CLIENT_ID"]
CLIENT_SECRET = os.environ["STRAVA_CLIENT_SECRET"]
REFRESH_TOKEN = os.environ["STRAVA_REFRESH_TOKEN"]
MONTHS_BACK = 36  # how many months of history to fetch
MAX_PER_PAGE = 100  # Strava max per page
PACE_FILTER = 15.0  # drop points with pace > this (min/km) — removes pauses
CACHE_FILE = "output/streams_cache.json"
# ─────────────────────────────────────────────────────────────────────────────


def pace_min_per_km(speed_ms):
    return 1000 / speed_ms / 60


def fmt_pace(p):
    m = int(p)
    s = int(round((p - m) * 60))
    return f"{m}:{s:02d}"


resp = requests.post(
    "https://www.strava.com/oauth/token",
    data={
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": REFRESH_TOKEN,
    },
)
resp.raise_for_status()
ACCESS_TOKEN = resp.json()["access_token"]

HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
BASE = "https://www.strava.com/api/v3"

# ── 1. Fetch all runs ─────────────────────────────────────────────────────────
after_dt = datetime.now(timezone.utc) - timedelta(days=MONTHS_BACK * 30)
after_ts = int(after_dt.timestamp())

print(f"Fetching runs since {after_dt.strftime('%b %Y')} ...")
all_runs, page = [], 1
while True:
    r = requests.get(
        f"{BASE}/athlete/activities",
        headers=HEADERS,
        params={"per_page": MAX_PER_PAGE, "page": page, "after": after_ts},
    )
    r.raise_for_status()
    batch = r.json()
    if not batch:
        break
    runs = [a for a in batch if a.get("type") == "Run" or a.get("sport_type") == "Run"]
    all_runs.extend(runs)
    if len(batch) < MAX_PER_PAGE:
        break
    page += 1

print(f"  Found {len(all_runs)} runs.\n")

# ── 2. Fetch streams for each run and bucket by week ──────────────────────────
streams_cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        streams_cache = json.load(f)
    print(f"Loaded {len(streams_cache)} cached streams from {CACHE_FILE}.\n")

weekly = defaultdict(lambda: {"hr": [], "pace": []})
cache_updated = False

for i, a in enumerate(all_runs):
    date = a["start_date_local"][:10]
    d = datetime.strptime(date, "%Y-%m-%d")
    iso = d.isocalendar()
    week = f"{iso[0]}-W{iso[1]:02d}"  # "YYYY-WXX", uses ISO year for Dec/Jan edge cases
    label = f"{date}  {a['distance']/1000:.1f} km"
    activity_id = str(a["id"])

    if not a.get("average_heartrate"):
        print(f"  [{i+1}/{len(all_runs)}] {label} — no HR, skipping")
        continue

    if activity_id in streams_cache:
        hr = streams_cache[activity_id]["hr"]
        vel = streams_cache[activity_id]["vel"]
    else:
        print(f"  [{i+1}/{len(all_runs)}] {label} — fetching streams ...")
        try:
            r = requests.get(
                f"{BASE}/activities/{activity_id}/streams",
                headers=HEADERS,
                params={"keys": "heartrate,velocity_smooth", "key_by_type": "true"},
            )
            r.raise_for_status()
            data = r.json()
            hr = data.get("heartrate", {}).get("data", [])
            vel = data.get("velocity_smooth", {}).get("data", [])
            streams_cache[activity_id] = {"hr": hr, "vel": vel}
            cache_updated = True
        except Exception as e:
            print(f"    → error: {e}")
            continue

    cnt = min(len(hr), len(vel))
    for j in range(cnt):
        if vel[j] > 0.5:  # moving
            p = pace_min_per_km(vel[j])
            if p <= PACE_FILTER:  # not paused/stopped
                weekly[week]["hr"].append(hr[j])
                weekly[week]["pace"].append(p)

if cache_updated:
    with open(CACHE_FILE, "w") as f:
        json.dump(streams_cache, f)
    print(f"Updated cache: {CACHE_FILE}\n")

if not weekly:
    print("No data collected. Check your token or date range.")
    exit()


## ── 3. Compute 95th percentiles per week ─────────────────────────────────────
weeks_sorted = sorted(weekly.keys())
results = []
for w in weeks_sorted:
    hr_vals = weekly[w]["hr"]
    pace_vals = weekly[w]["pace"]
    if len(hr_vals) < 150:
        print(f"  Skipping {w} — too few data points ({len(hr_vals)})")
        continue
    results.append(
        {
            "week": w,
            "label": f"W{int(w.split('-W')[1])} '{w[:4][2:]}",
            "hr_p95": np.percentile(hr_vals, 95),
            "pace_p10": np.percentile(pace_vals, 10),  # 10th = fastest end
            "hr_med": np.median(hr_vals),
            "pace_med": np.median(pace_vals),
            "hr_mean": np.mean(hr_vals),
            "pace_mean": np.mean(pace_vals),
            "n_pts": len(hr_vals),
        }
    )

if not results:
    print("Not enough data per week to plot.")
    exit()

print(f"\nWeekly summary ({len(results)} weeks):")
print(f"{'Week':<10} {'P95 HR':>8} {'P10 Pace':>10} {'Points':>8}")
for r in results:
    print(
        f"  {r['label']:<10} {r['hr_p95']:>6.1f} bpm   "
        f"{fmt_pace(r['pace_p10']):>8} /km   {r['n_pts']:>6,}"
    )


# ── 4. Colour map: oldest = cool blue, newest = warm orange ──────────────────
n = len(results)
cmap = plt.get_cmap("plasma", n)
colors = [cmap(i / max(n - 1, 1)) for i in range(n)]


def make_scatter(ax, x_key, y_key, xlabel, ylabel, title, results, colors, sizes):
    """Generic scatter panel with connecting line, labels, and colorbar."""
    xs = [r[x_key] for r in results]
    ys = [r[y_key] for r in results]

    # Connecting line
    for i in range(len(xs) - 1):
        ax.plot(
            [xs[i], xs[i + 1]],
            [ys[i], ys[i + 1]],
            color="white",
            linewidth=0.8,
            alpha=0.25,
            zorder=1,
        )

    ax.scatter(
        xs,
        ys,
        c=colors,
        s=sizes,
        zorder=3,
        edgecolors="white",
        linewidths=0.8,
        alpha=0.92,
    )

    for i, r in enumerate(results):
        ax.annotate(
            r["label"],
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
            color=colors[i],
            fontweight="bold",
            zorder=4,
        )

    ax.set_xlabel(xlabel, color="white", fontsize=10)
    ax.set_ylabel(ylabel, color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.set_facecolor("#0f1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(color="#333", linewidth=0.6, alpha=0.7)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)

    # Format y-axis as MM:SS pace
    ax.figure.canvas.draw()
    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([fmt_pace(y) if y > 0 else "" for y in y_ticks], color="white")

    # ↙ improving arrow in bottom-left
    ax.annotate(
        "",
        xy=(min(xs) - 0.8, min(ys) - 0.12),
        xytext=(min(xs) + 1.2, min(ys) + 0.22),
        arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.1),
    )
    ax.text(
        min(xs) + 1.4,
        min(ys) + 0.25,
        "improving →",
        color="#aaa",
        fontsize=7.5,
        va="bottom",
    )

    return xs, ys


# ── 5. Two-panel scatter: P95 (top) and Median (bottom) ──────────────────────
fig, (ax_p95, ax_med) = plt.subplots(2, 1, figsize=(12, 13))
fig.patch.set_facecolor("#0f1117")

sizes = [max(120, min(500, r["n_pts"] / 8)) for r in results]

make_scatter(
    ax_p95,
    x_key="hr_p95",
    y_key="pace_p10",
    xlabel="95th Percentile Heart Rate (bpm)",
    ylabel="10th Percentile Pace (min/km)",
    title="P95 HR vs P10 Pace — peak HR at fastest speeds",
    results=results,
    colors=colors,
    sizes=sizes,
)

make_scatter(
    ax_med,
    x_key="hr_med",
    y_key="pace_med",
    xlabel="Median Heart Rate (bpm)",
    ylabel="Median Pace (min/km)",
    title="Median — reflects typical easy/aerobic running",
    results=results,
    colors=colors,
    sizes=sizes,
)

# Shared colorbar
sm = cm.ScalarMappable(cmap="plasma", norm=mcolors.Normalize(vmin=0, vmax=n - 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax_p95, ax_med], pad=0.02, fraction=0.02)
cbar.set_ticks([0, n - 1])
cbar.set_ticklabels([results[0]["label"], results[-1]["label"]])
cbar.ax.yaxis.set_tick_params(color="white")
cbar.outline.set_edgecolor("#444")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=9)
cbar.set_label("Time →", color="white", fontsize=9)

fig.suptitle(
    "Running Fitness Over Time\n" "↙ = lower HR at faster pace = improved fitness",
    color="white",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

fig.savefig(
    "output/strava_weekly_scatter.png",
    dpi=180,
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
)
print("\nSaved: output/strava_weekly_scatter.png")


# ── 6. Bar charts: P95 vs Median for HR and Pace ─────────────────────────────
fig2, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
fig2.patch.set_facecolor("#0f1117")
(ax_hr_p95, ax_hr_med), (ax_pa_p95, ax_pa_med) = axes

xlabels = [r["label"] for r in results]
x = np.arange(len(results))
bar_w = 0.65


def style_bar_ax(ax):
    ax.set_facecolor("#111")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.tick_params(colors="white")
    ax.grid(axis="y", color="#333", linewidth=0.6, alpha=0.7)


for ax in axes.flat:
    style_bar_ax(ax)

# HR — P95
bars = ax_hr_p95.bar(
    x,
    [r["hr_p95"] for r in results],
    color=colors,
    edgecolor="#222",
    linewidth=0.5,
    width=bar_w,
)
ax_hr_p95.set_ylabel("Heart Rate (bpm)", color="white", fontsize=10)
ax_hr_p95.set_title(
    "95th Percentile HR\n(hard efforts ceiling)",
    color="white",
    fontsize=10,
    fontweight="bold",
)
# HR — Median
bars = ax_hr_med.bar(
    x,
    [r["hr_med"] for r in results],
    color=colors,
    edgecolor="#222",
    linewidth=0.5,
    width=bar_w,
)
ax_hr_med.set_title(
    "Median HR\n(typical aerobic effort)", color="white", fontsize=10, fontweight="bold"
)
# Pace — P95
bars = ax_pa_p95.bar(
    x,
    [r["pace_p10"] for r in results],
    color=colors,
    edgecolor="#222",
    linewidth=0.5,
    width=bar_w,
)
ax_pa_p95.set_ylabel("Pace (min/km)", color="white", fontsize=10)
ax_pa_p95.set_title(
    "10th Percentile Pace\n(fastest efforts)",
    color="white",
    fontsize=10,
    fontweight="bold",
)
tick_step = 4
ax_pa_p95.set_xticks(x[::tick_step])
ax_pa_p95.set_xticklabels(xlabels[::tick_step], rotation=45, ha="right", color="white", fontsize=8)
y_ticks = ax_pa_p95.get_yticks()
ax_pa_p95.set_yticks(y_ticks)
ax_pa_p95.set_yticklabels(
    [fmt_pace(y) if y > 0 else "" for y in y_ticks], color="white"
)

# Pace — Median
bars = ax_pa_med.bar(
    x,
    [r["pace_med"] for r in results],
    color=colors,
    edgecolor="#222",
    linewidth=0.5,
    width=bar_w,
)
ax_pa_med.set_title(
    "Median Pace\n(typical easy running)", color="white", fontsize=10, fontweight="bold"
)
ax_pa_med.set_xticks(x[::tick_step])
ax_pa_med.set_xticklabels(xlabels[::tick_step], rotation=45, ha="right", color="white", fontsize=8)
y_ticks = ax_pa_med.get_yticks()
ax_pa_med.set_yticks(y_ticks)
ax_pa_med.set_yticklabels(
    [fmt_pace(y) if y > 0 else "" for y in y_ticks], color="white"
)

fig2.suptitle(
    "Weekly HR & Pace — 95th Percentile vs Median",
    color="white",
    fontsize=13,
    fontweight="bold",
)
fig2.tight_layout()
fig2.savefig("output/strava_weekly_bars.png", dpi=180, facecolor=fig2.get_facecolor())
print("Saved: output/strava_weekly_bars.png")

plt.show()
print("\nDone!")
