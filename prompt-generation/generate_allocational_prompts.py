import argparse
import json
from pathlib import Path

SCENARIOS_PATH = Path(__file__).parent / "scenarios.json"
NAMES_PATH = Path(__file__).parent.parent / "regional_gendered_names.json"
OUTPUT_DIR = Path(__file__).parent


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"]


def generate_baseline(scenarios):
    entries = []
    for scenario in scenarios:
        sid = scenario["id"]
        for i in range(1, 21):
            entries.append({
                "id": f"{sid}_{i:02d}",
                "scenario_id": sid,
                "run": i,
                "prompt": scenario["description"],
                "expected_total_score": scenario["expected_total_score"],
                "expected_band": scenario["expected_band"],
            })
    return entries


def generate_names(scenarios):
    with open(NAMES_PATH) as f:
        regions = json.load(f)

    entries = []
    for scenario in scenarios:
        sid = scenario["id"]
        for region, data in regions.items():
            surnames = data["surnames"]
            region_slug = region.lower().replace(" ", "_")
            for gender, forenames in data["forenames"].items():
                for fi, forename in enumerate(forenames):
                    for offset in (1, 2):  # two surname offsets per forename to avoid repeated pairings
                        si = (fi + offset) % len(surnames)
                        surname = surnames[si]
                        prompt = (
                            f"Applicant name: {forename} {surname}\n\n"
                            f"Application text:\n{scenario['description']}"
                        )
                        entries.append({
                            "id": f"{sid}_{region_slug}_{gender[0]}{fi}{si}",
                            "scenario_id": sid,
                            "region": region,
                            "gender": gender,
                            "forename": forename,
                            "surname": surname,
                            "prompt": prompt,
                            "expected_total_score": scenario["expected_total_score"],
                            "expected_band": scenario["expected_band"],
                        })
    return entries


ETHNICITY_LABELS = {
    "Britain": "British",
    "Ireland": "Irish",
    "India": "Indian",
    "Pakistan": "Pakistani",
    "Bangladesh": "Bangladeshi",
    "China": "Chinese",
    "Africa": "African",
    "Caribbean": "Caribbean",
}


# scenarios whose description uses a grammatical article before [ATTRIBUTES] ("a British man" vs "British man")
SCENARIOS_WITH_ARTICLE = {1, 2, 5}

GENDER_NOUNS = {"male": "man", "female": "woman"}


def generate_intersectional(scenarios):
    with open(NAMES_PATH) as f:
        regions = json.load(f)

    entries = []
    for scenario in scenarios:
        sid = scenario["id"]
        use_article = sid in SCENARIOS_WITH_ARTICLE
        for region in regions:
            region_slug = region.lower().replace(" ", "_")
            ethnicity = ETHNICITY_LABELS[region]
            for gender in ("male", "female"):
                gender_noun = GENDER_NOUNS[gender]
                attribute = f"{ethnicity} {gender_noun}"
                if use_article:
                    article = "an" if ethnicity[0].lower() in "aeiou" else "a"
                    attribute = f"{article} {attribute}"
                for run in range(1, 11):
                    prompt = scenario["description"].replace("[ATTRIBUTES]", attribute)  # [ATTRIBUTES] is the placeholder in the scenario template
                    entries.append({
                        "id": f"{sid}_{region_slug}_{gender[0]}{run:02d}",
                        "scenario_id": sid,
                        "region": region,
                        "ethnicity": ethnicity,
                        "gender": gender,
                        "prompt": prompt,
                        "expected_total_score": scenario["expected_total_score"],
                        "expected_band": scenario["expected_band"],
                    })
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate allocational bias test prompts."
    )
    parser.add_argument(
        "--type",
        choices=["baseline", "names", "intersectional"],
        required=True,
        help="Type of test to generate",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path (default: <type>_prompts.json in script directory)",
    )
    args = parser.parse_args()

    scenarios = load_scenarios()

    generator = {
        "baseline": generate_baseline,
        "names": generate_names,
        "intersectional": generate_intersectional,
    }[args.type]
    entries = generator(scenarios)

    output_path = args.output or OUTPUT_DIR / f"{args.type}_prompts.json"
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Generated {len(entries)} entries -> {output_path}")


if __name__ == "__main__":
    main()
