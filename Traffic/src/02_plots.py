"""
02 - Exploratory Plots & Congestion Analysis
PeMS-BAY Dataset — SF Bay Area Highway Traffic
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

BASE = r'/Users/andrewsmac/CCIR/Traffic'
DATA = os.path.join(BASE, 'data', 'PEMS-BAY.csv')
OUT  = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)

# Set global style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'text.color': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'grid.color': 'lightgray',
    'font.family': 'sans-serif',
    'font.size': 10,
})

print("=" * 65)
print("  02 -- EXPLORATORY PLOTS & CONGESTION ANALYSIS")
print("=" * 65)

# ── Load ──
print("\n[1/5] Loading data...")
df = pd.read_csv(DATA)
df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
sensors = df.columns.tolist()
print(f"    Loaded {df.shape[0]:,} x {df.shape[1]}")

CONG_THRESH = 60.0

# ═══════════════════════════════════════════════════════
# PLOT 1: Speed distribution + hourly profile
# ═══════════════════════════════════════════════════════
print("\n[2/5] Plotting speed distribution & hourly profile...")
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 1a: Distribution
all_spd = df.values.flatten()
all_spd = all_spd[~np.isnan(all_spd)]
axes[0].hist(all_spd, bins=120, color='#58A6FF', alpha=0.85, edgecolor='none', density=True)
axes[0].axvline(x=CONG_THRESH, color='#F85149', linestyle='--', lw=2, label=f'Congestion threshold ({CONG_THRESH} mph)')
axes[0].set_title('Speed Distribution — All Sensors, All Time', fontweight='bold', fontsize=13, color='black')
axes[0].set_xlabel('Speed (mph)')
axes[0].set_ylabel('Density')
axes[0].legend(facecolor='white', edgecolor='black')

# 1b: Hourly profile
hourly = df.groupby(df.index.hour).mean()
hmean = hourly.mean(axis=1)
hstd  = hourly.std(axis=1)
hmin  = hourly.min(axis=1)
hmax  = hourly.max(axis=1)
axes[1].fill_between(hmean.index, hmin, hmax, alpha=0.15, color='#F85149', label='Min-Max range')
axes[1].fill_between(hmean.index, hmean - hstd, hmean + hstd, alpha=0.3, color='#58A6FF', label='Mean +/- 1 std')
axes[1].plot(hmean.index, hmean.values, color='#58A6FF', lw=2.5, marker='o', ms=5, label='Network mean')
axes[1].axhline(y=CONG_THRESH, color='#F85149', linestyle='--', lw=1.5, alpha=0.7)
axes[1].axvspan(7, 9, alpha=0.08, color='#FFA657', label='AM Rush')
axes[1].axvspan(16, 18, alpha=0.08, color='#FF7B72', label='PM Rush')
axes[1].set_title('Average Speed by Hour of Day', fontweight='bold', fontsize=13, color='black')
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Speed (mph)')
axes[1].set_xticks(range(0, 24))
axes[1].legend(facecolor='white', edgecolor='black', fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'plot1_speed_distribution.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("    -> Saved plot1_speed_distribution.png")

# ═══════════════════════════════════════════════════════
# PLOT 2: Day-of-week x Hour heatmap
# ═══════════════════════════════════════════════════════
print("\n[3/5] Plotting congestion heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 2a: Speed heatmap
tmp = df.copy()
tmp['hour'] = tmp.index.hour
tmp['dow']  = tmp.index.dayofweek
heat = tmp.groupby(['dow', 'hour'])[sensors].mean().mean(axis=1).unstack()
sns.heatmap(heat, cmap='RdYlGn', ax=axes[0], cbar_kws={'label': 'Avg Speed (mph)'},
            linewidths=0.3, linecolor='lightgray')
axes[0].set_title('Speed: Day of Week x Hour', fontweight='bold', fontsize=13, color='black')
axes[0].set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
axes[0].set_xlabel('Hour')

# 2b: Congestion probability heatmap
cong_mask = df < CONG_THRESH
cong_heat = cong_mask.copy()
cong_heat['hour'] = cong_heat.index.hour
cong_heat['dow']  = cong_heat.index.dayofweek
cong_pct = cong_heat.groupby(['dow', 'hour'])[sensors].mean().mean(axis=1).unstack() * 100
sns.heatmap(cong_pct, cmap='YlOrRd', ax=axes[1], cbar_kws={'label': '% Sensors Congested'},
            linewidths=0.3, linecolor='lightgray')
axes[1].set_title('Congestion Probability: Day x Hour', fontweight='bold', fontsize=13, color='black')
axes[1].set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
axes[1].set_xlabel('Hour')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'plot2_heatmaps.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("    -> Saved plot2_heatmaps.png")

# ═══════════════════════════════════════════════════════
# PLOT 3: Bottleneck identification
# ═══════════════════════════════════════════════════════
print("\n[4/5] Identifying bottlenecks & stress areas...")

congestion_freq = cong_mask.mean() * 100
congestion_sorted = congestion_freq.sort_values(ascending=False)

FREE_FLOW = df.quantile(0.95)
severity = ((FREE_FLOW - df.where(cong_mask)).clip(lower=0)).mean()
peak_mask = df.index.hour.isin([7, 8, 9, 16, 17, 18])
peak_cong = cong_mask.loc[peak_mask].mean() * 100

stress = (congestion_freq / 100) * (severity / severity.max()) * (peak_cong / 100)
stress = (stress / stress.max() * 100).round(1)
stress_sorted = stress.sort_values(ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Bottleneck & Stress Area Identification', fontsize=16, fontweight='bold', color='black', y=0.98)

# 3a: Top 20 congested
top20 = congestion_sorted.head(20)
colors = ['#F85149' if v > 50 else '#FFA657' if v > 35 else '#E3B341' for v in top20.values]
axes[0, 0].barh(range(len(top20)), top20.values, color=colors, edgecolor='none', height=0.7)
axes[0, 0].set_yticks(range(len(top20)))
axes[0, 0].set_yticklabels([f"S-{s}" for s in top20.index], fontsize=8)
axes[0, 0].set_xlabel('% Time Congested (<60 mph)')
axes[0, 0].set_title('Top 20 Most Congested Sensors', fontweight='bold', fontsize=12, color='black')
axes[0, 0].invert_yaxis()
for i, v in enumerate(top20.values):
    axes[0, 0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=7, color='black')

# 3b: Stress score distribution
bins_edge = [0, 10, 25, 50, 75, 100.1]
labels = ['Low\n(0-10)', 'Moderate\n(10-25)', 'High\n(25-50)', 'Very High\n(50-75)', 'Critical\n(75-100)']
cat = pd.cut(stress, bins=bins_edge, labels=labels, include_lowest=True)
counts = cat.value_counts().reindex(labels).fillna(0).astype(int)
bar_colors = ['#3FB950', '#E3B341', '#FFA657', '#F85149', '#DA3633']
bars = axes[0, 1].bar(range(len(counts)), counts.values, color=bar_colors, edgecolor='none', width=0.65)
axes[0, 1].set_xticks(range(len(counts)))
axes[0, 1].set_xticklabels(labels, fontsize=9)
axes[0, 1].set_title('Sensor Stress Level Distribution', fontweight='bold', fontsize=12, color='black')
axes[0, 1].set_ylabel('Number of Sensors')
for bar, v in zip(bars, counts.values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(v), ha='center', fontweight='bold', fontsize=10, color='black')

# 3c: Rush vs non-rush scatter
rush_avg = df.loc[peak_mask].mean()
nonrush_avg = df.loc[~peak_mask].mean()
sc = axes[1, 0].scatter(nonrush_avg, rush_avg, alpha=0.6, c=stress, cmap='hot_r', s=25, edgecolor='none')
axes[1, 0].plot([45, 75], [45, 75], '--', color='#484F58', lw=1)
axes[1, 0].set_xlabel('Non-Rush Hour Avg Speed (mph)')
axes[1, 0].set_ylabel('Rush Hour Avg Speed (mph)')
axes[1, 0].set_title('Rush vs Non-Rush Speed per Sensor', fontweight='bold', fontsize=12, color='black')
plt.colorbar(sc, ax=axes[1, 0], label='Stress Score', shrink=0.8)

# 3d: Top 5 bottleneck hourly congestion profile
top5 = stress_sorted.head(5).index.tolist()
cmap5 = ['#F85149', '#FFA657', '#E3B341', '#58A6FF', '#BC8CFF']
for sid, c in zip(top5, cmap5):
    h_cong = cong_mask[sid].groupby(df.index.hour).mean() * 100
    axes[1, 1].plot(h_cong.index, h_cong.values, marker='o', lw=2, ms=5, color=c, label=f'S-{sid}')
axes[1, 1].axhspan(0, 15, alpha=0.05, color='#3FB950')
axes[1, 1].axhspan(50, 100, alpha=0.05, color='#F85149')
axes[1, 1].set_title('Congestion by Hour — Top 5 Bottlenecks', fontweight='bold', fontsize=12, color='black')
axes[1, 1].set_xlabel('Hour of Day')
axes[1, 1].set_ylabel('% Time Congested')
axes[1, 1].set_xticks(range(0, 24))
axes[1, 1].legend(facecolor='white', edgecolor='black', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'plot3_bottlenecks.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("    -> Saved plot3_bottlenecks.png")

# ═══════════════════════════════════════════════════════
# PLOT 4: Time-series deep dive & weekly pattern
# ═══════════════════════════════════════════════════════
print("\n[5/5] Time-series deep dive...")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Time-Series Deep Dive', fontsize=16, fontweight='bold', color='black', y=0.98)

# 4a: Network-wide daily congestion trend
daily_cong = cong_mask.mean(axis=1).resample('D').mean() * 100
axes[0, 0].fill_between(daily_cong.index, daily_cong.values, alpha=0.35, color='#F85149')
axes[0, 0].plot(daily_cong.index, daily_cong.values, color='#F85149', lw=1)
axes[0, 0].set_title('Network-Wide Daily Congestion', fontweight='bold', fontsize=12, color='black')
axes[0, 0].set_ylabel('% Sensors Congested')
axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
axes[0, 0].xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=30)

# 4b: Worst sensor 1-week detail
worst = stress_sorted.index[0]
week = df[worst].loc['2017-03-06':'2017-03-12']
axes[0, 1].fill_between(week.index, week.values, alpha=0.3, color='#58A6FF')
axes[0, 1].plot(week.index, week.values, color='#58A6FF', lw=0.8)
axes[0, 1].axhline(y=CONG_THRESH, color='#F85149', ls='--', lw=1.5, alpha=0.7)
axes[0, 1].set_title(f'Worst Bottleneck (S-{worst}) — 1 Week', fontweight='bold', fontsize=12, color='black')
axes[0, 1].set_ylabel('Speed (mph)')
axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=30, fontsize=7)

# 4c: Weekday vs weekend for worst sensor
wd_mask = df.index.dayofweek < 5
wd_hr = df[worst].loc[wd_mask].groupby(df.loc[wd_mask].index.hour).mean()
we_hr = df[worst].loc[~wd_mask].groupby(df.loc[~wd_mask].index.hour).mean()
axes[1, 0].plot(wd_hr.index, wd_hr.values, 'o-', color='#F85149', lw=2, ms=6, label='Weekday')
axes[1, 0].plot(we_hr.index, we_hr.values, 'o-', color='#58A6FF', lw=2, ms=6, label='Weekend')
axes[1, 0].axhline(y=CONG_THRESH, color='#484F58', ls='--', lw=1)
axes[1, 0].set_title(f'S-{worst}: Weekday vs Weekend', fontweight='bold', fontsize=12, color='black')
axes[1, 0].set_xlabel('Hour')
axes[1, 0].set_ylabel('Avg Speed (mph)')
axes[1, 0].legend(facecolor='white', edgecolor='black')
axes[1, 0].set_xticks(range(0, 24))

# 4d: Sensor ranking bar
sensor_means = df.mean().sort_values()
bar_c = ['#F85149' if v < 58 else '#FFA657' if v < 62 else '#E3B341' if v < 65 else '#3FB950' for v in sensor_means.values]
axes[1, 1].bar(range(len(sensor_means)), sensor_means.values, color=bar_c, width=1.0, edgecolor='none')
axes[1, 1].axhline(y=CONG_THRESH, color='#F85149', ls='--', lw=1.5, alpha=0.7)
axes[1, 1].set_title('All Sensors Ranked by Average Speed', fontweight='bold', fontsize=12, color='black')
axes[1, 1].set_xlabel('Sensor Rank')
axes[1, 1].set_ylabel('Avg Speed (mph)')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'plot4_timeseries.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("    -> Saved plot4_timeseries.png")

# ── Save bottleneck rankings ──
bneck = pd.DataFrame({
    'sensor_id': sensors,
    'avg_speed': df.mean().values,
    'min_speed': df.min().values,
    'std_speed': df.std().values,
    'congestion_pct': congestion_freq.values,
    'peak_congestion_pct': peak_cong.values,
    'severity_mph_drop': severity.values,
    'stress_score': stress.values
}).sort_values('stress_score', ascending=False)
bneck.to_csv(os.path.join(OUT, 'bottleneck_rankings.csv'), index=False)
print("\n    -> Saved bottleneck_rankings.csv")

print("\n" + "=" * 65)
print("  ALL PLOTS GENERATED SUCCESSFULLY")
print("=" * 65)
