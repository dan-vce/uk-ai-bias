import argparse
from pathlib import Path
from together import Together

READY_TO_SUBMIT = Path(__file__).parent / "ready_to_submit"

parser = argparse.ArgumentParser(description="Submit a Together AI batch job from a JSONL file.")
parser.add_argument("input_file", help="Filename within the ready_to_submit folder (e.g. batch_input_together.jsonl)")
args = parser.parse_args()

input_file: Path = READY_TO_SUBMIT / args.input_file
batch_id_file: Path = Path(__file__).parent / "batch_id_together.txt"

client = Together()

file_response = client.files.upload(file=str(input_file), purpose="batch-api", check=False)
file_id = file_response.id
print(f"Uploaded file: {file_id}")

# .create() returns a wrapper object; the batch is at .job
response = client.batches.create(
    input_file_id=file_id,
    endpoint="/v1/chat/completions",
)
batch = response.job

batch_id_file.write_text(batch.id)

print(f"Batch submitted: {batch.id}")
print(f"Batch ID saved to {batch_id_file}")
print(f"Run retrieve_batch_together.py when you're ready to download results.")
