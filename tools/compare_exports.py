"""Regression check: compare a fresh export / evaluation against the thesis reference.

1) Export-level (optional, needs the original thesis export directory):
     python tools/compare_exports.py exports --new comparative_eval/mardm_3way_additive \
         --ref /path/to/thesis/mardm_3way_additive_5styles [--modes styled transfer]
   Per-sample max |diff| of feat67 and joints, plus how many samples are bit-identical.

2) Metric-level:
     python tools/compare_exports.py metrics --new comparative_eval/results.json \
         --ref comparative_eval/reference/thesis_results_main.json --model mardm_3way_additive
   Prints new vs. thesis values and flags deviations beyond a relative tolerance.
   SRA is reported but not gated: the thesis used a 20-class classifier, the release ships 21 classes.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

GATED = ['FID', 'R_precision_top_1', 'R_precision_top_3', 'RR_MPJPE', 'Global_MPJPE', 'Skating', 'Diversity']


def cmp_exports(args):
    worst = 0.0
    for mode in args.modes:
        new_dir, ref_dir = os.path.join(args.new, mode), os.path.join(args.ref, mode)
        files = sorted(glob.glob(os.path.join(ref_dir, '*.npy')))
        if not files:
            print(f"[{mode}] no reference files in {ref_dir}")
            continue
        diffs, missing, identical, shape_mismatch = [], 0, 0, 0
        for f in files:
            g = os.path.join(new_dir, os.path.basename(f))
            if not os.path.exists(g):
                missing += 1
                continue
            a, b = np.load(g), np.load(f)
            if a.shape != b.shape:
                shape_mismatch += 1
                continue
            d = float(np.abs(a - b).max()) if a.size else 0.0
            identical += d == 0.0
            diffs.append(d)
        diffs = np.array(diffs) if diffs else np.array([np.nan])
        worst = max(worst, np.nanmax(diffs))
        print(f"[{mode}] files {len(files)}  missing {missing}  shape-mismatch {shape_mismatch}  "
              f"bit-identical {identical}  max|diff| median {np.nanmedian(diffs):.3g}  max {np.nanmax(diffs):.3g}")
    print("RESULT:", "IDENTICAL" if worst == 0 else f"differs (max |diff| {worst:.3g}); judge with the metric check")


def cmp_metrics(args):
    new, ref = json.load(open(args.new)), json.load(open(args.ref))
    fails = 0
    for mode in ('base', 'styled', 'transfer'):
        r = ref.get(mode, {}).get(args.ref_model or args.model)
        n = new.get(mode, {}).get(args.model)
        if r is None or n is None:
            continue
        print(f"\n[{mode}]  {'metric':22s} {'thesis':>10s} {'new':>10s} {'rel.diff':>9s}")
        for k, rv in r.items():
            if not isinstance(rv, (int, float)) or k.startswith('SRA_T') or k not in n:
                continue
            nv = n[k]
            rel = abs(nv - rv) / max(abs(rv), 1e-8)
            gated = k in GATED
            bad = gated and rel > args.rtol
            fails += bad
            tag = 'FAIL' if bad else ('' if gated else '(not gated)')
            print(f"        {k:22s} {rv:10.4f} {nv:10.4f} {rel:9.1%} {tag}")
    print(f"\nRESULT: {'PASS' if fails == 0 else f'{fails} gated metric(s) outside ±{args.rtol:.0%}'}")
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('exports')
    e.add_argument('--new', required=True)
    e.add_argument('--ref', required=True)
    e.add_argument('--modes', nargs='+', default=['styled', 'transfer', 'base'])
    m = sub.add_parser('metrics')
    m.add_argument('--new', required=True)
    m.add_argument('--ref', default='comparative_eval/reference/thesis_results_main.json')
    m.add_argument('--model', default='mardm_3way_additive')
    m.add_argument('--ref_model', default=None, help='model key in the reference file (default: --model)')
    m.add_argument('--rtol', type=float, default=0.05, help='relative tolerance for gated metrics (default 5%%)')
    args = ap.parse_args()
    if args.cmd == 'exports':
        cmp_exports(args)
    else:
        sys.exit(1 if cmp_metrics(args) else 0)


if __name__ == '__main__':
    main()
