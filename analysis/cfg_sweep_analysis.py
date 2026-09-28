import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import sys

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'text.usetex': False,
})

# ============================================================
# Load data
# ============================================================
json_path = sys.argv[1] if len(sys.argv) > 1 else 'results_cfg_sweep.json'
with open(json_path, 'r') as f:
    data = json.load(f)

styled = data['styled']
transfer = data['transfer']

STYLES = ['Aeroplane', 'Chicken', 'Robot', 'Superman', 'ArmsFolded']

# ============================================================
# Parse sweep models and extract s_style values
# ============================================================
# Model keys look like: mardm_3way_t4.5_s1.0, mardm_3way_t4.5_s2.5, etc.
def parse_s_style(key):
    """Extract s_style float from model key like mardm_3way_t4.5_s2.5"""
    parts = key.split('_s')
    if len(parts) == 2:
        try:
            return float(parts[1])
        except ValueError:
            return None
    return None

# Find all sweep models (exclude 'gt')
sweep_keys_styled = sorted(
    [k for k in styled if k != 'gt' and parse_s_style(k) is not None],
    key=lambda k: parse_s_style(k)
)
sweep_keys_transfer = sorted(
    [k for k in transfer if parse_s_style(k) is not None],
    key=lambda k: parse_s_style(k)
)

s_vals_styled = [parse_s_style(k) for k in sweep_keys_styled]
s_vals_transfer = [parse_s_style(k) for k in sweep_keys_transfer]

# ============================================================
# Extract metrics from JSON
# ============================================================
def extract_series(section, keys, metric):
    return [section[k].get(metric, 0.0) for k in keys]

styled_sra1 = extract_series(styled, sweep_keys_styled, 'SRA_top_1')
styled_rp3 = [styled[k]['R_precision_top_3'] * 100 for k in sweep_keys_styled]
styled_fid = extract_series(styled, sweep_keys_styled, 'FID')
styled_div = extract_series(styled, sweep_keys_styled, 'Diversity')
styled_skt = extract_series(styled, sweep_keys_styled, 'Skating')

trans_sra1 = extract_series(transfer, sweep_keys_transfer, 'SRA_top_1')
trans_rp3 = [transfer[k]['R_precision_top_3'] * 100 for k in sweep_keys_transfer]
trans_div = extract_series(transfer, sweep_keys_transfer, 'Diversity')
trans_skt = extract_series(transfer, sweep_keys_transfer, 'Skating')

# GT reference values
gt = styled.get('gt', {})
gt_sra1 = gt.get('SRA_top_1', 91.67)
gt_rp3 = gt.get('R_precision_top_3', 0.531) * 100
gt_div = gt.get('Diversity', 7.06)
gt_skt = gt.get('Skating', 0.178)

# Per-style SRA for each sweep point
perstyle_styled = {}
for style in STYLES:
    key = f'SRA_T1_{style}'
    perstyle_styled[style] = [styled[k].get(key, 0.0) for k in sweep_keys_styled]

# ============================================================
# PLOT: 2x2 panel
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

c_styled = '#377eb8'
c_transfer = '#e41a1c'

# ---- Panel (a): SRA₁ and R-Prec₃ ----
ax = axes[0, 0]
ln1 = ax.plot(s_vals_styled, styled_sra1, 'o-', color=c_styled,
              label='SRA$_1$ (styled)', linewidth=2, markersize=5)
ln2 = ax.plot(s_vals_transfer, trans_sra1, 's--', color=c_transfer,
              label='SRA$_1$ (transfer)', linewidth=2, markersize=5)

ax2 = ax.twinx()
ln3 = ax2.plot(s_vals_styled, styled_rp3, 'o-', color=c_styled, alpha=0.4,
               label='R-Prec$_3$ (styled)', linewidth=1.5, markersize=4)
ln4 = ax2.plot(s_vals_transfer, trans_rp3, 's--', color=c_transfer, alpha=0.4,
               label='R-Prec$_3$ (transfer)', linewidth=1.5, markersize=4)

ax.axhline(y=gt_sra1, color='grey', ls=':', lw=0.8, label='GT SRA$_1$')
ax2.axhline(y=gt_rp3, color='grey', ls='-.', lw=0.8)

ax.set_xlabel('$s_{style}$')
ax.set_ylabel('SRA$_1$ [%] (solid)')
ax2.set_ylabel('R-Prec$_3$ [%] (faded)')
ax.set_title('(a) Style fidelity vs. content preservation')
ax.set_ylim(20, 100)
ax2.set_ylim(40, 100)

lns = ln1 + ln2 + ln3 + ln4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='center left', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.15)

# ---- Panel (b): FID ----
ax = axes[0, 1]
ax.plot(s_vals_styled, styled_fid, 'o-', color=c_styled,
        label='Styled', linewidth=2, markersize=5)

min_idx = np.argmin(styled_fid)
ax.scatter(s_vals_styled[min_idx], styled_fid[min_idx], s=150,
           facecolors='none', edgecolors=c_styled, linewidths=2, zorder=10)
ax.annotate(f'min = {styled_fid[min_idx]:.2f}\n($s_{{style}}$ = {s_vals_styled[min_idx]})',
            xy=(s_vals_styled[min_idx], styled_fid[min_idx]),
            xytext=(s_vals_styled[min_idx] + 1.2, styled_fid[min_idx] + 0.3),
            fontsize=8, color=c_styled,
            arrowprops=dict(arrowstyle='->', color=c_styled, lw=1))

ax.set_xlabel('$s_{style}$')
ax.set_ylabel('FID $\\downarrow$')
ax.set_title('(b) Distributional quality')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)

# ---- Panel (c): Diversity ----
ax = axes[1, 0]
ax.plot(s_vals_styled, styled_div, 'o-', color=c_styled,
        label='Styled', linewidth=2, markersize=5)
ax.plot(s_vals_transfer, trans_div, 's--', color=c_transfer,
        label='Transfer', linewidth=2, markersize=5)
ax.axhline(y=gt_div, color='grey', ls=':', lw=0.8, label=f'GT = {gt_div:.2f}')

ax.set_xlabel('$s_{style}$')
ax.set_ylabel('Diversity $\\rightarrow$')
ax.set_title('(c) Output diversity')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)

# ---- Panel (d): Skating ----
ax = axes[1, 1]
ax.plot(s_vals_styled, styled_skt, 'o-', color=c_styled,
        label='Styled', linewidth=2, markersize=5)
ax.plot(s_vals_transfer, trans_skt, 's--', color=c_transfer,
        label='Transfer', linewidth=2, markersize=5)
ax.axhline(y=gt_skt, color='grey', ls=':', lw=0.8, label=f'GT = {gt_skt:.3f}')

ax.set_xlabel('$s_{style}$')
ax.set_ylabel('Skating Ratio $\\downarrow$')
ax.set_title('(d) Physical plausibility')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)

plt.tight_layout(h_pad=2.5, w_pad=2)
plt.savefig('plot_cfg_sweep.pdf')
plt.savefig('plot_cfg_sweep.png')
plt.close()
print("CFG sweep plot saved.")

# ============================================================
# Per-style SRA₁ vs s_style
# ============================================================
style_colors = {
    'Aeroplane': '#d62728',
    'Chicken': '#ff7f0e',
    'Robot': '#2ca02c',
    'Superman': '#1f77b4',
    'ArmsFolded': '#9467bd',
}

fig, ax = plt.subplots(figsize=(8, 4.5))

for style in STYLES:
    ax.plot(s_vals_styled, perstyle_styled[style], 'o-',
            color=style_colors[style], label=style,
            linewidth=2, markersize=5)

ax.set_xlabel('$s_{style}$ (text scale fixed at $s_{text} = 4.5$)')
ax.set_ylabel('SRA$_1$ [%]')
ax.set_title('Per-Style Sensitivity to Style Guidance Scale (Styled Generation)')
ax.set_ylim(-5, 105)
ax.legend(loc='center right', framealpha=0.9)
ax.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig('plot_cfg_perstyle.pdf')
plt.savefig('plot_cfg_perstyle.png')
plt.close()
print("Per-style CFG sweep plot saved.")

print(f"\nDone. Loaded from: {json_path}")
print("Usage: python cfg_sweep_plots_v2.py [path/to/results_cfg_sweep.json]")