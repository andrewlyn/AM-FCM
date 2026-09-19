from pathlib import Path

from pathlib import Path

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / 'results_feature_combinations_global_snr' / 'feature_combination_summary.csv')

sns.set_theme(style='whitegrid', context='paper')
fig, ax = plt.subplots(figsize=(8, 5))
piv = df[df.feature_set == 'M'].pivot(index='snr_db', columns='frequency_hz', values='le5_pct').sort_index()
freqs = sorted(piv.columns)
pal = sns.color_palette('viridis', n_colors=len(freqs))
for f, c in zip(freqs, pal):
    ax.plot(piv.index, piv[f], marker='o', markersize=5, linewidth=1.8, label=f'{f} Hz', color=c)
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('P(|err| <= 5ms) (%)')
ax.set_title('Feature M: le5 by dominant frequency (non-monotonicity source)', fontweight='bold')
ax.set_xticks(sorted(piv.index))
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(title='Dominant Freq.', fontsize=9)
plt.tight_layout()
plt.savefig(BASE + r'\fig_le5_by_frequency.png', dpi=300, bbox_inches='tight')
plt.close()
print('fig1 saved')

order = ['M', 'P', 'Std', 'M+P', 'M+Std', 'P+Std', 'M+P+Std']
snrs = [-10.0, -6.0, 0.0, 6.0]
agg = df[df.feature_set.isin(order)].groupby(['snr_db', 'feature_set'])[
    ['le2_pct', 'le5_pct', 'le10_pct']].mean().reset_index()
rows = []
for s in snrs:
    for f in order:
        r = agg[(agg.snr_db == s) & (agg.feature_set == f)].iloc[0]
        rows.append({'snr_db': s, 'feature_set': f,
                     'le10_pct': round(r.le10_pct, 1),
                     'le5_pct': round(r.le5_pct, 1),
                     'le2_pct': round(r.le2_pct, 1)})
out = pd.DataFrame(rows)
wide = out.pivot(index='feature_set', columns='snr_db', values=['le10_pct', 'le5_pct', 'le2_pct'])
new_cols = []
for m, s in wide.columns:
    new_cols.append('{:+.0f}dB_{}'.format(s, m.replace('_pct', '')))
wide.columns = new_cols
wide = wide.reindex(order)
wide.to_csv(BASE + r'\table_typical_snr.csv', encoding='utf-8-sig')
print('table saved')
print(wide.round(1).to_string())
