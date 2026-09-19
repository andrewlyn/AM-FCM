import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = 'feature_combination_summary.csv'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, CSV_FILE)

TARGET_SETS = ['M', 'P', 'Std', 'M+P', 'M+Std', 'P+Std', 'M+P+Std']

CATEGORY_MAP = {'M': 'Single', 'P': 'Single', 'Std': 'Single',
                'M+P': 'Pair', 'M+Std': 'Pair', 'P+Std': 'Pair',
                'M+P+Std': 'Triple'}

METRICS = ['le2_pct', 'le5_pct', 'le10_pct']
METRIC_LABELS = {'le2_pct': 'P(|err| ≤ 2ms)', 'le5_pct': 'P(|err| ≤ 5ms)',
                 'le10_pct': 'P(|err| ≤ 10ms)'}

sns.set_theme(style='whitegrid', context='paper')

df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
print(f'[1] Raw data: {len(df)} rows, {len(df.columns)} columns')

df_sel = df[df['feature_set'].isin(TARGET_SETS)].copy()
print(f'[2] Selected {TARGET_SETS}; total {len(df_sel)} rows '
      f'(dominant frequencies {sorted(df_sel["frequency_hz"].unique())} Hz, '
      f'SNR {sorted(df_sel["snr_db"].unique())} dB)')

df_agg = (df_sel.groupby(['snr_db', 'feature_set'], as_index=False)[METRICS]
              .mean()
              .round(2))
df_agg['category'] = df_agg['feature_set'].map(CATEGORY_MAP)

set_order = {s: i for i, s in enumerate(TARGET_SETS)}
df_agg = (df_agg.sort_values(['snr_db', 'category', 'feature_set'],
                             key=lambda s: s.map(set_order) if s.name == 'feature_set' else s)
                .reset_index(drop=True))

print(f'[3] Frequency averaging complete: {len(df_agg)} rows '
      f'(= {df_agg["snr_db"].nunique()} SNR values × {df_agg["feature_set"].nunique()} feature combinations)')

long_path = os.path.join(BASE_DIR, 'mps_agg_by_snr.csv')
df_agg[METRICS] = df_agg[METRICS].round(2)
df_agg.to_csv(long_path, index=False, encoding='utf-8-sig')
print(f'[4a] Long-format table saved: {long_path}')

wide = df_agg.pivot(index='snr_db', columns='feature_set', values=METRICS)
wide = wide.reindex(columns=pd.MultiIndex.from_product([METRICS, TARGET_SETS]))
wide.columns = wide.columns.swaplevel(0, 1)  # (feature_set, metric)
wide = wide.reindex(columns=pd.MultiIndex.from_product([TARGET_SETS, METRICS]))
wide.columns = [f'{feat}_{m.replace("_pct", "")}' for feat, m in wide.columns]
wide = wide.sort_index().round(2)
wide_path = os.path.join(BASE_DIR, 'mps_wide_by_snr.csv')
wide.to_csv(wide_path, encoding='utf-8-sig')
print(f'[4b] Wide-format table saved: {wide_path}')

print('\n' + '=' * 90)
print('Hit rates (%) averaged across dominant frequencies at each SNR; rows = feature sets, columns = thresholds')
print('=' * 90)
for snr in sorted(df_agg['snr_db'].unique()):
    sub = df_agg[df_agg['snr_db'] == snr].set_index('feature_set')
    sub = sub.reindex(TARGET_SETS)[METRICS]
    print(f'\n--- SNR = {snr:+.1f} dB ---')
    print(sub.round(2).to_string())

CMAP = sns.color_palette('viridis', as_cmap=True)

n_snr = df_agg['snr_db'].nunique()
n_set = len(TARGET_SETS)
fig, axes = plt.subplots(1, 3, figsize=(18, max(5, n_snr * 0.55)))
fig.suptitle('M / P / Std Combinations: Hit Rate Heatmap (SNR x Feature Set)',
             fontsize=14, fontweight='bold', y=1.02)
for ax, m in zip(axes, METRICS):
    piv = df_agg.pivot(index='snr_db', columns='feature_set', values=m)
    piv = piv.reindex(columns=TARGET_SETS).sort_index(ascending=False)
    sns.heatmap(piv, annot=True, fmt='.1f', cmap=CMAP, cbar_kws={'label': '%'},
                ax=ax, linewidths=0.5, annot_kws={'fontsize': 8})
    ax.set_title(METRIC_LABELS[m], fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Combination', fontsize=10)
    ax.set_ylabel('SNR (dB)', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
plt.tight_layout()
heatmap_path = os.path.join(BASE_DIR, 'mps_performance_heatmaps.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'[5a] Heatmap saved: {heatmap_path}')

palette = sns.color_palette('tab10', n_colors=len(TARGET_SETS))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('M / P / Std Combinations: Hit Rate vs SNR (Frequency-Averaged)',
             fontsize=14, fontweight='bold', y=1.02)
for ax, m in zip(axes, METRICS):
    sns.lineplot(data=df_agg, x='snr_db', y=m, hue='feature_set',
                 hue_order=TARGET_SETS, palette=palette, ax=ax,
                 linewidth=1.8, alpha=0.9, markers=True, markersize=6)
    ax.set_title(METRIC_LABELS[m], fontsize=12, fontweight='bold')
    ax.set_xlabel('SNR (dB)', fontsize=11)
    ax.set_ylabel('Hit Rate (%)', fontsize=11)
    ax.set_xticks(sorted(df_agg['snr_db'].unique()))
    ax.grid(True, linestyle='--', alpha=0.6)
    if m == METRICS[-1]:
        ax.legend(title='Feature Set', bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=9, title_fontsize=10)
    else:
        ax.get_legend().remove()
plt.tight_layout(rect=[0, 0, 0.90, 0.96])
lines_path = os.path.join(BASE_DIR, 'mps_performance_lines.png')
plt.savefig(lines_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'[5b] Line plot saved: {lines_path}')

print('\n[Complete] All results have been generated.')
