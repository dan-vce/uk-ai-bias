import argparse
import json
import re
from pathlib import Path

SCENARIOS_PATH = Path(__file__).parent / "prompt-generation" / "allocational" / "scenarios.json"


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        data = json.load(f)
    return {s["id"]: s for s in data["scenarios"]}


def extract_json_from_text(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1), strict=False)
    return json.loads(text, strict=False)


def extract_openai(line):
    custom_id = line["custom_id"]
    output = line["response"]["body"]["output"]
    message = next(item for item in output if item["type"] == "message")
    text = message["content"][0]["text"]
    return custom_id, extract_json_from_text(text)


def extract_anthropic(line):
    custom_id = line["custom_id"]
    content = line["result"]["message"]["content"]
    text = next(block["text"] for block in content if block["type"] == "text")
    return custom_id, extract_json_from_text(text)


def extract_together(line):
    custom_id = line["custom_id"]
    text = line["response"]["body"]["choices"][0]["message"]["content"]
    return custom_id, extract_json_from_text(text)


def extract_google(line):
    custom_id = line["custom_id"]
    if "error" in line:
        raise ValueError(line["error"])
    text = line["response"]["text"]
    return custom_id, extract_json_from_text(text)


_LABEL_TO_BAND = {
    "emergency": "A",
    "critical": "B",
    "high priority": "C",
    "medium priority": "D",
    "low priority": "E",
}

def normalize_band(band_str):
    # handles: 'Band B (High Priority)', bare 'B', or label like 'Critical'
    match = re.search(r"Band\s+([A-E])", band_str, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if band_str.strip().upper() in {"A", "B", "C", "D", "E"}:
        return band_str.strip().upper()
    label = _LABEL_TO_BAND.get(band_str.strip().lower())
    if label:
        return label
    return band_str


def build_entry(custom_id, model_response, scenario):
    # custom_id format: "{scenario_id}_{run}"
    scenario_id, run = (int(x) for x in custom_id.split("_"))

    expected_score = scenario["expected_total_score"]
    expected_band = scenario["expected_band"]
    expected_breakdown = scenario["expected_scoring_breakdown"]

    model_band = normalize_band(model_response.get("priority_band", ""))
    model_score = model_response.get("total_points")
    model_breakdown = model_response.get("scoring_breakdown", {})

    breakdown_diff = {
        key: model_breakdown.get(key, 0) - expected_breakdown.get(key, 0)
        for key in expected_breakdown
    }

    return {
        "custom_id": custom_id,
        "scenario_id": scenario_id,
        "run": run,
        "model_response": model_response,
        "expected": {
            "total_score": expected_score,
            "band": expected_band,
            "scoring_breakdown": expected_breakdown,
        },
        "variance": {
            "total_points_diff": model_score - expected_score if model_score is not None else None,
            "band_match": model_band == expected_band,
            "calculation_accurate": model_score == sum(model_breakdown.values()) if model_score is not None else None,  # model's own arithmetic, not vs ground truth
            "scoring_breakdown_diff": breakdown_diff,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Parse baseline batch results and join with scenario metadata.")
    parser.add_argument("input_file", type=Path, help="Path to the results JSONL file")
    parser.add_argument(
        "--api",
        choices=["openai", "anthropic", "together", "google"],
        required=True,
        help="Which API produced the results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: <input_stem>_parsed.json next to input)",
    )
    args = parser.parse_args()

    output_file = args.output or args.input_file.parent / f"{args.input_file.stem}_parsed.json"
    extractor = {"openai": extract_openai, "anthropic": extract_anthropic, "together": extract_together, "google": extract_google}[args.api]
    scenarios = load_scenarios()

    entries = []
    errors = []
    with args.input_file.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            raw = json.loads(line)
            try:
                custom_id, model_response = extractor(raw)
                scenario_id = int(custom_id.split("_")[0])
                entry = build_entry(custom_id, model_response, scenarios[scenario_id])
                entries.append(entry)
            except Exception as e:
                errors.append({"line": lineno, "custom_id": raw.get("custom_id"), "error": str(e)})

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    print(f"Parsed {len(entries)} entries -> {output_file}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for err in errors:
            print(f"  Line {err['line']} ({err['custom_id']}): {err['error']}")


if __name__ == "__main__":
    main()
