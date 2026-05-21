import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Convert a prompts JSON file to OpenAI batch JSONL.")
parser.add_argument(
    "input_file",
    type=Path,
    help="Path to the input prompts JSON file (e.g. baseline_prompts.json)",
)
parser.add_argument(
    "system_prompt_file",
    type=Path,
    help="Path to a .txt file whose contents will be used as the system message",
)
args = parser.parse_args()

input_file: Path = args.input_file
system_prompt_file: Path = args.system_prompt_file
output_file: Path = Path(__file__).parent / "ready_to_submit" / "batch_input_openai.jsonl"

MODEL = "gpt-5.4-mini"
MAX_OUTPUT_TOKENS = 3000

def create_batch_file():
    system_msg = system_prompt_file.read_text(encoding="utf-8").strip()
    entries = json.loads(input_file.read_text(encoding="utf-8"))

    with output_file.open("w", encoding="utf-8", newline="") as f:
        for entry in entries:
            record = {
                "custom_id": str(entry["id"]),
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": MODEL,
                    "input": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": entry["prompt"]}
                    ],
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "reasoning": {"effort": "low"}
                }
            }
            f.write(json.dumps(record) + "\n")
    print(f"Created {output_file} with {len(entries)} entries.")

if __name__ == "__main__":
    create_batch_file()