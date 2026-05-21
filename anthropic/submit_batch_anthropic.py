import json
import argparse
from pathlib import Path
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

READY_TO_SUBMIT = Path(__file__).parent / "ready_to_submit"

parser = argparse.ArgumentParser(description="Submit an Anthropic batch job from a JSONL file.")
parser.add_argument("input_file", help="Filename within the ready_to_submit folder (e.g. batch_input_anthropic.jsonl)")
args = parser.parse_args()

input_file: Path = READY_TO_SUBMIT / args.input_file
batch_id_file: Path = Path(__file__).parent / "batch_id_anthropic.txt"

client = anthropic.Anthropic()

requests = []
with input_file.open("r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        requests.append(
            Request(
                custom_id=item["custom_id"],
                params=MessageCreateParamsNonStreaming(**item["params"])
            )
        )

batch_job = client.messages.batches.create(requests=requests)

batch_id_file.write_text(batch_job.id)

print(f"Batch submitted: {batch_job.id}")
print(f"Batch ID saved to {batch_id_file}")
print(f"Run retrieve_batch_anthropic.py when you're ready to download results (within 29 days).")
