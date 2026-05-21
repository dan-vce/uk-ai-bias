import argparse
import json
from pathlib import Path
from google import genai

READY_TO_SUBMIT = Path(__file__).parent / "ready_to_submit"

parser = argparse.ArgumentParser(description="Submit a Google Gemini batch job from a JSONL file.")
parser.add_argument("input_file", help="Filename within the ready_to_submit folder (e.g. batch_input_google.jsonl)")
args = parser.parse_args()

input_file: Path = READY_TO_SUBMIT / args.input_file
batch_name_file: Path = Path(__file__).parent / "batch_id_google.txt"
custom_ids_file: Path = Path(__file__).parent / "batch_custom_ids_google.json"

MODEL = "gemini-3-flash-preview"

client = genai.Client()

custom_ids = []
requests = []
with input_file.open("r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        custom_ids.append(item["custom_id"])
        requests.append({"contents": item["contents"], "config": item["config"]})

batch_job = client.batches.create(model=MODEL, src=requests)

# Gemini batch API doesn't echo custom IDs — save separately to reconstruct results
batch_name_file.write_text(batch_job.name)
custom_ids_file.write_text(json.dumps(custom_ids))

print(f"Batch submitted: {batch_job.name}")
print(f"Batch name saved to {batch_name_file}")
print(f"Custom ID mapping saved to {custom_ids_file}")
print(f"Run retrieve_batch_google.py when you're ready to download results.")
