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
    'legend.fontsize': 9.5,
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
json_path = sys.argv[1] if len(sys.argv) > 1 else 'results.json'
with open(json_path, 'r') as f:
    data = json.load(f)

styled = data['styled']
transfer = data['transfer']

MODEL_LABELS = {
    'gt': 'GT',
    'smoodi': 'SMooDi',
    'loramdm': 'LoRA-MDM',
    'mardm_2way': 'VISTA-2way',
    'mardm_3way_additive': 'VISTA-3way',
}

STYLES = ['Aeroplane', 'Chicken', 'Robot', 'Superman', 'ArmsFolded']
STYLES_SHORT = ['Aero.', 'Chicken', 'Robot', 'Super.', 'ArmsF.']

COLORS = {
    'GT': '#2d2d2d',
    'SMooDi': '#4daf4a',
    'LoRA-MDM': '#ff7f00',
    'VISTA-2way': '#377eb8',
    'VISTA-3way': '#e41a1c',
}

MARKERS = {
    'GT': '*',
    'SMooDi': 's',
    'LoRA-MDM': '^',
    'VISTA-2way': 'o',
    'VISTA-3way': 'D',
}

SIZES = {
    'GT': 200,
    'SMooDi': 100,
    'LoRA-MDM': 100,
    'VISTA-2way': 120,
    'VISTA-3way': 120,
}


def get_label(key):
    return MODEL_LABELS.get(key, key)


def get_perstyle_sra(section, model):
    d = section[model]
    return [d.get(f'SRA_T1_{s}', 0.0) for s in STYLES]


# ============================================================
# PLOT B: Content-Style Tradeoff Scatter
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

styled_order = ['gt', 'smoodi', 'loramdm', 'mardm_2way', 'mardm_3way_additive']
styled_points = {}
for m in styled_order:
    if m not in styled:
        continue
    label = get_label(m)
    rp3 = styled[m]['R_precision_top_3'] * 100
    sra1 = styled[m]['SRA_top_1']
    styled_points[label] = {'rp3': rp3, 'sra1': sra1}
    ax1.scatter(rp3, sra1, c=COLORS[label], marker=MARKERS[label],
                s=SIZES[label], label=label, zorder=5,
                edgecolors='black', linewidths=0.5)

if 'VISTA-2way' in styled_points and 'VISTA-3way' in styled_points:
    p2, p3 = styled_points['VISTA-2way'], styled_points['VISTA-3way']
    ax1.annotate('', xy=(p3['rp3'], p3['sra1']),
                 xytext=(p2['rp3'], p2['sra1']),
                 arrowprops=dict(arrowstyle='->', color='#888888',
                                 lw=1.5, connectionstyle='arc3,rad=-0.15'))
    ax1.annotate('$s_{style}$\n$4.5 \\rightarrow 2.0$',
                 xy=((p2['rp3']+p3['rp3'])/2+1, (p2['sra1']+p3['sra1'])/2+3),
                 fontsize=8, color='#555555', ha='center')

if 'GT' in styled_points:
    ax1.axhline(y=styled_points['GT']['sra1'], color='#cccccc', ls='--', lw=0.8, zorder=1)
    ax1.axvline(x=styled_points['GT']['rp3'], color='#cccccc', ls='--', lw=0.8, zorder=1)

n_styled = styled.get('gt', {}).get('num_samples', '?')
ax1.set_xlabel('R-Precision (Top-3) [%]  →  content preservation')
ax1.set_ylabel('SRA$_1$ [%]  →  style fidelity')
ax1.set_title(f'(a) Styled Generation ($n$={n_styled})')
ax1.set_xlim(30, 95)
ax1.set_ylim(-5, 100)
ax1.legend(loc='lower left', framealpha=0.9)
ax1.grid(True, alpha=0.15)

transfer_order = ['smoodi', 'loramdm', 'mardm_2way', 'mardm_3way_additive']
transfer_points = {}
for m in transfer_order:
    if m not in transfer:
        continue
    label = get_label(m)
    rp3 = transfer[m]['R_precision_top_3'] * 100
    sra1 = transfer[m]['SRA_top_1']
    transfer_points[label] = {'rp3': rp3, 'sra1': sra1}
    ax2.scatter(rp3, sra1, c=COLORS[label], marker=MARKERS[label],
                s=SIZES[label], label=label, zorder=5,
                edgecolors='black', linewidths=0.5)

if 'VISTA-2way' in transfer_points and 'VISTA-3way' in transfer_points:
    p2, p3 = transfer_points['VISTA-2way'], transfer_points['VISTA-3way']
    ax2.annotate('', xy=(p3['rp3'], p3['sra1']),
                 xytext=(p2['rp3'], p2['sra1']),
                 arrowprops=dict(arrowstyle='->', color='#888888',
                                 lw=1.5, connectionstyle='arc3,rad=-0.15'))
    ax2.annotate('$s_{style}$\n$4.5 \\rightarrow 2.0$',
                 xy=((p2['rp3']+p3['rp3'])/2+1, (p2['sra1']+p3['sra1'])/2+5),
                 fontsize=8, color='#555555', ha='center')

n_transfer = list(transfer.values())[0].get('num_samples', '?') if transfer else '?'
ax2.set_xlabel('R-Precision (Top-3) [%]  →  content preservation')
ax2.set_title(f'(b) Zero-Shot Transfer ($n$={n_transfer})')
ax2.set_xlim(50, 82)
ax2.set_ylim(-5, 100)
ax2.legend(loc='lower left', framealpha=0.9)
ax2.grid(True, alpha=0.15)

plt.tight_layout(w_pad=2)
plt.savefig('plot_b_tradeoff.pdf')
plt.savefig('plot_b_tradeoff.png')
plt.close()
print("Plot B saved.")

# ============================================================
# PLOT A: Per-Style SRA₁ Bar Chart
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

styled_methods = [m for m in styled_order if m in styled]
n_methods = len(styled_methods)
x = np.arange(len(STYLES))
width = 0.15
offsets = np.arange(n_methods) - (n_methods - 1) / 2

for i, m in enumerate(styled_methods):
    label = get_label(m)
    vals = get_perstyle_sra(styled, m)
    ax1.bar(x + offsets[i]*width, vals, width*0.9,
            label=label, color=COLORS[label],
            edgecolor='white', linewidth=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels(STYLES_SHORT)
ax1.set_ylabel('SRA$_1$ [%]')
ax1.set_title(f'(a) Styled Generation ($n$={n_styled})')
ax1.set_ylim(0, 110)
ax1.axhline(y=100, color='#cccccc', ls='--', lw=0.5)
ax1.legend(loc='upper left', ncol=2, fontsize=8.5, framealpha=0.9)
ax1.grid(axis='y', alpha=0.15)

transfer_methods = [m for m in transfer_order if m in transfer]
n_methods_t = len(transfer_methods)
offsets_t = np.arange(n_methods_t) - (n_methods_t - 1) / 2
width_t = 0.18

for i, m in enumerate(transfer_methods):
    label = get_label(m)
    vals = get_perstyle_sra(transfer, m)
    ax2.bar(x + offsets_t[i]*width_t, vals, width_t*0.9,
            label=label, color=COLORS[label],
            edgecolor='white', linewidth=0.5)

ax2.set_xticks(x)
ax2.set_xticklabels(STYLES_SHORT)
ax2.set_ylabel('SRA$_1$ [%]')
ax2.set_title(f'(b) Zero-Shot Transfer ($n$={n_transfer})')
ax2.set_ylim(0, 110)
ax2.axhline(y=100, color='#cccccc', ls='--', lw=0.5)
ax2.legend(loc='upper left', ncol=2, fontsize=8.5, framealpha=0.9)
ax2.grid(axis='y', alpha=0.15)

plt.tight_layout(w_pad=2)
plt.savefig('plot_a_perstyle.pdf')
plt.savefig('plot_a_perstyle.png')
plt.close()
print("Plot A saved.")

print(f"\nDone. Loaded from: {json_path}")
print("Usage: python thesis_plots_v2.py [path/to/results.json]")