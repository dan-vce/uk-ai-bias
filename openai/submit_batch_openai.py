import argparse
from pathlib import Path
from openai import OpenAI

READY_TO_SUBMIT = Path(__file__).parent / "ready_to_submit"

parser = argparse.ArgumentParser(description="Submit an OpenAI batch job from a JSONL file.")
parser.add_argument("input_file", help="Filename within the ready_to_submit folder (e.g. batch_input_openai.jsonl)")
args = parser.parse_args()

input_file: Path = READY_TO_SUBMIT / args.input_file
batch_id_file: Path = Path(__file__).parent / "batch_id_openai.txt"

client = OpenAI()

file_upload = client.files.create(
    file=input_file.open("rb"),
    purpose="batch"
)

batch_job = client.batches.create(
    input_file_id=file_upload.id,
    endpoint="/v1/responses",
    completion_window="24h"
)

batch_id_file.write_text(batch_job.id)

print(f"Batch submitted: {batch_job.id}")
print(f"Batch ID saved to {batch_id_file}")
print(f"Run retrieve_batch_openai.py when you're ready to download results (within 29 days).")
