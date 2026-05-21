import argparse
import json
import re
from pathlib import Path

SCENARIOS_PATH = Path(__file__).parent / "prompt-generation" / "allocational" / "scenarios.json"
DEFAULT_PROMPTS_PATH = Path(__file__).parent / "prompt-generation" / "allocational" / "names_prompts.json"


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        data = json.load(f)
    return {s["id"]: s for s in data["scenarios"]}


def load_prompts(prompts_path):
    with open(prompts_path) as f:
        entries = json.load(f)
    return {e["id"]: e for e in entries}


def extract_json_from_text(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1), strict=False)
    return json.loads(text, strict=False)


def partial_extract(text):
    # model hit its token limit mid-response; JSON is incomplete so we pull fields out with regex
    result = {}

    m = re.search(r'"total_points"\s*:\s*(-?\d+)', text)
    if m:
        result["total_points"] = int(m.group(1))

    m = re.search(r'"priority_band"\s*:\s*"([^"]+)"', text)
    if m:
        result["priority_band"] = m.group(1)

    breakdown = {}
    for key in ("medical_points", "overcrowding_points", "homelessness_points", "community_contribution", "local_connection"):
        m = re.search(rf'"{key}"\s*:\s*(-?\d+)', text)
        if m:
            breakdown[key] = int(m.group(1))
    if breakdown:
        result["scoring_breakdown"] = breakdown

    m = re.search(r'"reasoning_summary"\s*:\s*"(.*)', text, re.DOTALL)
    partial_reasoning = m.group(1) if m else ""
    result["reasoning_summary"] = partial_reasoning + "[TOKEN LIMIT REACHED]"
    return result


def try_extract_json(text):
    try:
        return extract_json_from_text(text), False
    except json.JSONDecodeError:
        return partial_extract(text), True


def extract_openai(line):
    custom_id = line["custom_id"]
    output = line["response"]["body"]["output"]
    message = next(item for item in output if item["type"] == "message")
    text = message["content"][0]["text"]
    return custom_id, *try_extract_json(text)


def extract_anthropic(line):
    custom_id = line["custom_id"]
    content = line["result"]["message"]["content"]
    text = next(block["text"] for block in content if block["type"] == "text")
    return custom_id, *try_extract_json(text)


def extract_together(line):
    custom_id = line["custom_id"]
    text = line["response"]["body"]["choices"][0]["message"]["content"]
    return custom_id, *try_extract_json(text)


def extract_google(line):
    custom_id = line["custom_id"]
    if "error" in line:
        raise ValueError(line["error"])
    text = line["response"]["text"]
    return custom_id, *try_extract_json(text)


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


def build_entry(custom_id, model_response, truncated, prompt_meta, scenario):
    expected_score = scenario["expected_total_score"]
    expected_band = scenario["expected_band"]
    expected_breakdown = scenario["expected_scoring_breakdown"]

    model_band = normalize_band(model_response.get("priority_band", ""))
    model_score = model_response.get("total_points")
    model_breakdown = model_response.get("scoring_breakdown", {})
    variance = {
        "total_points_diff": model_score - expected_score if model_score is not None else None,
        "band_match": model_band == expected_band if model_band else None,
        "calculation_accurate": model_score == sum(model_breakdown.values()) if model_score is not None else None,  # model's own arithmetic, not vs ground truth
        "scoring_breakdown_diff": {
            # None means the criterion was absent from the response entirely, not just scored 0
            key: model_breakdown.get(key, 0) - expected_breakdown.get(key, 0)
            if key in model_breakdown else None
            for key in expected_breakdown
        },
    }

    return {
        "custom_id": custom_id,
        "scenario_id": prompt_meta["scenario_id"],
        "region": prompt_meta["region"],
        "gender": prompt_meta["gender"],
        "forename": prompt_meta["forename"],
        "surname": prompt_meta["surname"],
        "truncated": truncated,
        "model_response": model_response,
        "expected": {
            "total_score": expected_score,
            "band": expected_band,
            "scoring_breakdown": expected_breakdown,
        },
        "variance": variance,
    }


def main():
    parser = argparse.ArgumentParser(description="Parse names batch results and join with prompt and scenario metadata.")
    parser.add_argument("input_file", type=Path, help="Path to the results JSONL file")
    parser.add_argument(
        "--api",
        choices=["openai", "anthropic", "together", "google"],
        required=True,
        help="Which API produced the results",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help=f"Path to the names prompts JSON file (default: {DEFAULT_PROMPTS_PATH})",
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
    prompts = load_prompts(args.prompts)

    entries = []
    errors = []
    truncated_count = 0
    with args.input_file.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            raw = json.loads(line)
            try:
                custom_id, model_response, truncated = extractor(raw)
                prompt_meta = prompts[custom_id]
                scenario = scenarios[prompt_meta["scenario_id"]]
                entry = build_entry(custom_id, model_response, truncated, prompt_meta, scenario)
                entries.append(entry)
                if truncated:
                    truncated_count += 1
            except Exception as e:
                errors.append({"line": lineno, "custom_id": raw.get("custom_id"), "error": str(e)})

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    print(f"Parsed {len(entries)} entries -> {output_file}")
    if truncated_count:
        print(f"Truncated responses: {truncated_count}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for err in errors:
            print(f"  Line {err['line']} ({err['custom_id']}): {err['error']}")


if __name__ == "__main__":
    main()
