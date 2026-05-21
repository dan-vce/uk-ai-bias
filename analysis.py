#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CRITERIA = ["medical_points", "overcrowding_points", "homelessness_points", "community_contribution", "local_connection"]


def load_entries(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def direction_counts(diffs: list[int]) -> dict:
    # diffs are signed (model_score minus expected), so >0 means the model overscored
    n = len(diffs)
    return {
        "pct_over":  round(100 * sum(1 for d in diffs if d > 0) / n, 1),
        "pct_exact": round(100 * sum(1 for d in diffs if d == 0) / n, 1),
        "pct_under": round(100 * sum(1 for d in diffs if d < 0) / n, 1),
    }


def summarise(diffs: list[int], band_matches: list[bool]) -> dict:
    if not diffs:
        return {"n": 0}
    arr = np.asarray(diffs, dtype=float)
    return {
        "n": len(arr),
        "mae": round(float(np.mean(np.abs(arr))), 2),
        "std": round(float(np.std(arr, ddof=1)), 2) if len(arr) > 1 else 0.0,
        "mean_signed_diff": round(float(np.mean(arr)), 2),
        "band_match_rate": round(float(np.mean(band_matches)), 4) if band_matches else None,
        **direction_counts(diffs),
    }


def group_summarise(entries: list[dict], key: str) -> dict:
    groups = defaultdict(lambda: {"diffs": [], "band_matches": []})
    for e in entries:
        g = str(e[key])
        d = e["variance"].get("total_points_diff")
        bm = e["variance"].get("band_match")
        if d is not None:
            groups[g]["diffs"].append(d)
        if bm is not None:
            groups[g]["band_matches"].append(bm)
    return {
        g: summarise(v["diffs"], v["band_matches"])
        for g, v in sorted(groups.items())
    }


def criterion_summarise(entries: list[dict]) -> dict:
    groups = defaultdict(list)
    for e in entries:
        bd = e["variance"].get("scoring_breakdown_diff", {})
        for k in CRITERIA:
            v = bd.get(k)
            if v is not None:
                groups[k].append(v)
    return {
        k: {
            "mean_signed_diff": round(float(np.mean(groups[k])), 2),
            "mae": round(float(np.mean(np.abs(groups[k]))), 2),
            "n": len(groups[k]),
        }
        for k in CRITERIA if groups[k]
    }


def condition_block(entries: list[dict], has_region: bool = False, has_gender: bool = False) -> dict:
    diffs = [e["variance"]["total_points_diff"] for e in entries
             if e["variance"].get("total_points_diff") is not None]
    band_matches = [e["variance"]["band_match"] for e in entries
                    if e["variance"].get("band_match") is not None]
    block = {
        "summary": summarise(diffs, band_matches),
        "by_scenario": group_summarise(entries, "scenario_id"),
        "by_criterion": criterion_summarise(entries),
    }
    # baseline has no demographic fields, so region/gender breakdowns are opt-in
    if has_region:
        block["by_region"] = group_summarise(entries, "region")
    if has_gender:
        block["by_gender"] = group_summarise(entries, "gender")
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",       type=Path, required=True)
    ap.add_argument("--names",          type=Path, default=None)
    ap.add_argument("--intersectional", type=Path, default=None)
    ap.add_argument("--output",         type=Path, required=True)
    ap.add_argument("--model",          type=str,  required=True)
    args = ap.parse_args()

    baseline = load_entries(args.baseline)
    names    = load_entries(args.names)
    inter    = load_entries(args.intersectional)

    out = {
        "model": args.model,
        "note": "Descriptive statistics pooled across all scenarios 1-5.",
        "conditions": {
            "baseline":       condition_block(baseline),
            "names":          condition_block(names, has_region=True, has_gender=True),
            "intersectional": condition_block(inter, has_region=True, has_gender=True),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== {args.model} — descriptive analysis ===")
    print(f"{'Condition':<18}{'n':>6}{'MAE':>10}{'Std':>10}{'Mean diff':>11}{'Band match':>12}{'%Over':>8}{'%Exact':>8}{'%Under':>8}")
    for cond in ("baseline", "names", "intersectional"):
        s = out["conditions"][cond]["summary"]
        if s.get("n", 0) == 0:
            continue
        bm = f"{s['band_match_rate']:.2%}" if s["band_match_rate"] is not None else "n/a"
        print(f"{cond:<18}{s['n']:>6}{s['mae']:>10.1f}{s['std']:>10.1f}{s['mean_signed_diff']:>+11.1f}{bm:>12}"
              f"{s.get('pct_over', 0):>7.1f}%{s.get('pct_exact', 0):>7.1f}%{s.get('pct_under', 0):>7.1f}%")

    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
