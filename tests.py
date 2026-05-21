#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats


# Scenario 5 excluded: baseline BMR was 0.40 vs 0.96 for other scenarios — rubric failure, not demographic signal
EXCLUDE_SCENARIOS_BIAS = {5}

# Q_A: does adding demographic identifiers change overall scoring accuracy? (all scenarios, absolute error)
# Q_B: do specific demographic groups receive systematically different scores? (scenario 5 excluded)


def load_entries(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def filter_bias(entries: list[dict]) -> list[dict]:
    return [e for e in entries if e.get("scenario_id") not in EXCLUDE_SCENARIOS_BIAS]


def cliffs_delta(x, y) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return None
    diff = x[:, None] - y[None, :]  # outer difference: every x[i] vs every y[j]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (len(x) * len(y)))


def mann_whitney_abs(a, b, label: str) -> dict:
    # absolute error so that over- and under-scoring in opposite directions don't cancel out
    if not a or not b:
        return {"label": label, "note": "insufficient data"}
    if len(set(a)) <= 1 and len(set(b)) <= 1 and a[0] == b[0]:
        return {"label": label, "note": "both distributions constant and identical",
                "cliffs_delta": 0.0}
    try:
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError as exc:
        return {"label": label, "note": f"test failed: {exc}"}
    return {
        "label": label,
        "test": "Mann-Whitney U on absolute total_points_diff",
        "U": round(float(U), 1),
        "p_value": round(float(p), 4),
        "cliffs_delta": round(cliffs_delta(b, a), 3),
        "n_baseline": len(a),
        "n_other": len(b),
        "significant": bool(p < 0.05),
    }


def condition_vs_baseline(baseline_entries, other_entries, label: str) -> dict:
    a = [abs(e["variance"]["total_points_diff"]) for e in baseline_entries
         if e["variance"].get("total_points_diff") is not None]
    b = [abs(e["variance"]["total_points_diff"]) for e in other_entries
         if e["variance"].get("total_points_diff") is not None]
    return mann_whitney_abs(a, b, label)


def chi2_bmr_condition(baseline_entries, other_entries, label: str) -> dict:
    def counts(entries):
        match = sum(1 for e in entries if e["variance"].get("band_match"))
        total = sum(1 for e in entries if e["variance"].get("band_match") is not None)
        return match, total

    base_match, base_total = counts(baseline_entries)
    other_match, other_total = counts(other_entries)

    if base_total == 0 or other_total == 0:
        return {"label": label, "note": "insufficient data"}

    contingency = np.array([
        [base_match,  base_total  - base_match],
        [other_match, other_total - other_match],
    ])
    chi2, p, dof, _ = stats.chi2_contingency(contingency, correction=False)
    n = base_total + other_total
    phi = float(np.sqrt(chi2 / n))

    return {
        "label": label,
        "test": "Chi-square on band_match x condition",
        "chi2": round(float(chi2), 3),
        "p_value": round(float(p), 4),
        "dof": int(dof),
        "phi": round(phi, 3),
        "significant": bool(p < 0.05),
        "bmr_baseline": round(base_match / base_total, 3),
        "bmr_marked":   round(other_match / other_total, 3),
        "n_baseline": base_total,
        "n_marked":   other_total,
    }


def gender_bias_mwu(entries: list[dict], label: str) -> dict:
    male   = [abs(e["variance"]["total_points_diff"]) for e in entries
              if e.get("gender") == "male"   and
              e["variance"].get("total_points_diff") is not None]
    female = [abs(e["variance"]["total_points_diff"]) for e in entries
              if e.get("gender") == "female" and
              e["variance"].get("total_points_diff") is not None]

    if not male or not female:
        return {"label": label, "note": "insufficient data"}

    try:
        U, p = stats.mannwhitneyu(female, male, alternative="two-sided")
    except ValueError as exc:
        return {"label": label, "note": f"test failed: {exc}"}

    return {
        "label": label,
        "test": "Mann-Whitney U on absolute total_points_diff",
        "U": round(float(U), 1),
        "p_value": round(float(p), 4),
        "cliffs_delta": round(cliffs_delta(female, male), 3),
        "n_male": len(male),
        "n_female": len(female),
        "significant": bool(p < 0.05),
        "median_abs_male":   round(float(np.median(male)),   1),
        "median_abs_female": round(float(np.median(female)), 1),
    }


def chi2_bmr_gender(entries: list[dict], label: str) -> dict:
    male_match   = sum(1 for e in entries if e.get("gender") == "male"   and e["variance"].get("band_match"))
    male_total   = sum(1 for e in entries if e.get("gender") == "male"   and e["variance"].get("band_match") is not None)
    female_match = sum(1 for e in entries if e.get("gender") == "female" and e["variance"].get("band_match"))
    female_total = sum(1 for e in entries if e.get("gender") == "female" and e["variance"].get("band_match") is not None)

    if male_total == 0 or female_total == 0:
        return {"label": label, "note": "insufficient data"}

    contingency = np.array([
        [male_match,   male_total   - male_match],
        [female_match, female_total - female_match],
    ])
    chi2, p, dof, _ = stats.chi2_contingency(contingency, correction=False)
    n = male_total + female_total
    phi = float(np.sqrt(chi2 / n))

    return {
        "label": label,
        "test": "Chi-square on band_match x gender",
        "chi2": round(float(chi2), 3),
        "p_value": round(float(p), 4),
        "dof": int(dof),
        "phi": round(phi, 3),
        "significant": bool(p < 0.05),
        "bmr_male":   round(male_match   / male_total,   3),
        "bmr_female": round(female_match / female_total, 3),
        "n_male":   male_total,
        "n_female": female_total,
    }


def _region_abs_groups(entries: list[dict]) -> dict[str, list]:
    # {region: [abs(total_points_diff), ...]} — shared by KW and pairwise tests
    groups = defaultdict(list)
    for e in entries:
        d = e["variance"].get("total_points_diff")
        if d is not None and e.get("region"):
            groups[e["region"]].append(abs(d))
    return groups


def kruskal_regions_abs(entries: list[dict]) -> dict:
    groups = _region_abs_groups(entries)
    group_names = sorted(groups)
    if len(group_names) < 2:
        return {"note": "fewer than 2 groups"}

    try:
        H, p = stats.kruskal(*[groups[g] for g in group_names])
    except ValueError as exc:
        return {"note": f"test failed: {exc}"}

    return {
        "test": "Kruskal-Wallis H on absolute total_points_diff",
        "H": round(float(H), 3),
        "p_value": round(float(p), 4),
        "significant": bool(p < 0.05),
        "groups": group_names,
        "median_abs": {g: round(float(np.median(groups[g])), 1) for g in group_names},
    }


def pairwise_regions_abs(entries: list[dict]) -> dict:
    groups = _region_abs_groups(entries)
    region_names = sorted(groups)
    pairs = list(combinations(region_names, 2))
    n_comparisons = len(pairs)
    significant_comparisons = []
    n_significant = 0

    # all 28 pairs tested; only those significant after Bonferroni correction are stored
    for a, b in pairs:
        try:
            U, p = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        except ValueError:
            continue
        p_bonf = min(float(p) * n_comparisons, 1.0)
        if p_bonf < 0.05:
            n_significant += 1
            significant_comparisons.append({
                "pair": f"{a}_vs_{b}",
                "U": round(float(U), 1),
                "p_value": round(float(p), 4),
                "p_bonferroni": round(p_bonf, 4),
                "cliffs_delta": round(cliffs_delta(groups[a], groups[b]), 3),
                "n_a": len(groups[a]),
                "n_b": len(groups[b]),
            })

    return {
        "n_total": n_comparisons,
        "correction": "Bonferroni",
        "n_significant": n_significant,
        "comparisons": significant_comparisons,
    }


def chi2_bmr_regional(entries: list[dict], label: str) -> dict:
    groups = defaultdict(lambda: [0, 0])  # [match, no_match]
    for e in entries:
        bm = e["variance"].get("band_match")
        region = e.get("region")
        if bm is not None and region:
            if bm:
                groups[region][0] += 1
            else:
                groups[region][1] += 1

    region_names = sorted(groups)
    if len(region_names) < 2:
        return {"label": label, "note": "fewer than 2 groups"}

    contingency = np.array([[groups[r][0], groups[r][1]] for r in region_names])
    try:
        chi2, p, dof, _ = stats.chi2_contingency(contingency, correction=False)
    except ValueError as exc:
        return {"label": label, "note": f"test failed: {exc}"}

    n = int(contingency.sum())
    k = min(contingency.shape)  # Cramér's V denominator uses min(rows, cols)
    cramers_v = float(np.sqrt(chi2 / (n * (k - 1)))) if k > 1 else 0.0
    bmr_by_region = {
        r: round(groups[r][0] / (groups[r][0] + groups[r][1]), 3)
        for r in region_names
    }

    return {
        "label": label,
        "test": "Chi-square on band_match x region",
        "chi2": round(float(chi2), 3),
        "p_value": round(float(p), 4),
        "dof": int(dof),
        "cramers_v": round(cramers_v, 3),
        "significant": bool(p < 0.05),
        "bmr_by_region": bmr_by_region,
    }


def bias_block(entries: list[dict], label: str) -> dict:
    filtered = filter_bias(entries)
    n_excluded = len(entries) - len(filtered)

    gender_mwu  = gender_bias_mwu(filtered, f"gender_{label}")
    gender_bmr  = chi2_bmr_gender(filtered, f"bmr_gender_{label}")
    kw          = kruskal_regions_abs(filtered)
    pw          = pairwise_regions_abs(filtered) if kw.get("significant") else \
                  "skipped — Kruskal-Wallis not significant"
    regional_bmr = chi2_bmr_regional(filtered, f"bmr_regional_{label}")

    return {
        "note": f"Scenario 5 excluded ({n_excluded} entries removed — rubric failure in baseline).",
        "gender": gender_mwu,
        "bmr_gender": gender_bmr,
        "regional": {
            "kruskal_wallis": kw,
            "pairwise": pw,
        },
        "bmr_regional": regional_bmr,
    }


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

    def make_condition(other, lbl):
        if not other:
            return None
        return {
            "mwu": condition_vs_baseline(baseline, other, lbl),
            "bmr": chi2_bmr_condition(baseline, other, f"{lbl}_bmr"),
        }

    out = {
        "model": args.model,
        "note": (
            "Question A (condition tests): MWU on absolute error + chi-square on BMR — "
            "does introducing demographics change error magnitude / band accuracy? "
            "All scenarios included. "
            "Question B (bias tests): MWU and KW on ABSOLUTE error (not signed) — "
            "do some groups receive higher MAE? Chi-square on BMR x group also included. "
            "Scenario 5 excluded from bias tests (rubric failure in baseline)."
        ),
        "condition_tests": {
            "names_vs_baseline":          make_condition(names, "names_vs_baseline"),
            "intersectional_vs_baseline": make_condition(inter, "intersectional_vs_baseline"),
        },
        "bias_tests": {
            "names":          bias_block(names, "names") if names else None,
            "intersectional": bias_block(inter, "intersectional") if inter else None,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== {args.model} — statistical tests ===")

    print("\n--- Question A: condition vs baseline (all scenarios) ---")
    for test_name, t in out["condition_tests"].items():
        if t is None:
            continue
        mwu = t.get("mwu", {})
        bmr = t.get("bmr", {})
        if mwu and "U" in mwu:
            sig = "*" if mwu["significant"] else ""
            print(f"  {test_name} MWU:  U={mwu['U']}, p={mwu['p_value']}{sig}, "
                  f"delta={mwu['cliffs_delta']:+.3f}")
        if bmr and "chi2" in bmr:
            sig = "*" if bmr["significant"] else ""
            print(f"  {test_name} BMR:  chi2={bmr['chi2']}, p={bmr['p_value']}{sig}, "
                  f"phi={bmr['phi']:.3f}  "
                  f"(baseline={bmr['bmr_baseline']:.3f} → marked={bmr['bmr_marked']:.3f})")

    print("\n--- Question B: demographic bias (absolute error, scenario 5 excluded) ---")
    for cond_label in ("names", "intersectional"):
        bt = out["bias_tests"].get(cond_label)
        if not bt:
            continue
        print(f"\n  [{cond_label}]  {bt.get('note', '')}")

        g = bt["gender"]
        if "U" in g:
            sig = "*" if g["significant"] else ""
            print(f"    Gender MWU:  p={g['p_value']}{sig}, delta={g['cliffs_delta']:+.3f}  "
                  f"(med |err| male={g['median_abs_male']:.1f}, female={g['median_abs_female']:.1f})")
        else:
            print(f"    Gender MWU:  {g.get('note')}")

        bg = bt["bmr_gender"]
        if "chi2" in bg:
            sig = "*" if bg["significant"] else ""
            print(f"    Gender BMR:  chi2={bg['chi2']}, p={bg['p_value']}{sig}, phi={bg['phi']:.3f}  "
                  f"(male={bg['bmr_male']:.3f}, female={bg['bmr_female']:.3f})")

        kw = bt["regional"]["kruskal_wallis"]
        if "H" in kw:
            sig = "*" if kw["significant"] else ""
            print(f"    Regional KW: H={kw['H']}, p={kw['p_value']}{sig}")
            print(f"    Median |err|: {kw['median_abs']}")
        else:
            print(f"    Regional KW: {kw.get('note')}")

        pw = bt["regional"]["pairwise"]
        if isinstance(pw, str):
            print(f"    Pairwise:    {pw}")
        else:
            print(f"    Pairwise:    {pw['n_significant']}/{pw['n_total']} significant after Bonferroni")
            for c in pw["comparisons"]:
                print(f"      {c['pair']}: p={c['p_value']}, p_bonf={c['p_bonferroni']}, "
                      f"delta={c['cliffs_delta']:+.3f}")

        br = bt["bmr_regional"]
        if "chi2" in br:
            sig = "*" if br["significant"] else ""
            print(f"    Regional BMR: chi2={br['chi2']}, p={br['p_value']}{sig}, "
                  f"V={br['cramers_v']:.3f}")
            print(f"    BMR by region: {br['bmr_by_region']}")

    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
