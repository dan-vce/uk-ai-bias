import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Convert a prompts JSON file to Together AI batch JSONL.")
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
output_file: Path = Path(__file__).parent / "ready_to_submit" / "batch_input_together.jsonl"

MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = 1500

def create_batch_file():
    system_msg = system_prompt_file.read_text(encoding="utf-8").strip()
    entries = json.loads(input_file.read_text(encoding="utf-8"))

    with output_file.open("w", encoding="utf-8", newline="") as f:
        for entry in entries:
            record = {
                "custom_id": str(entry["id"]),
                "body": {
                    "model": MODEL,
                    "max_tokens": MAX_TOKENS,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": entry["prompt"]}
                    ]
                }
            }
            f.write(json.dumps(record) + "\n")
    print(f"Created {output_file} with {len(entries)} entries.")

if __name__ == "__main__":
    create_batch_file()
