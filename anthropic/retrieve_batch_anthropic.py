import json
from pathlib import Path
import anthropic

SCRIPT_DIR = Path(__file__).parent
client = anthropic.Anthropic()

batch_id_file = SCRIPT_DIR / "batch_id_anthropic.txt"
batch_id = batch_id_file.read_text().strip()
print(f"Checking batch: {batch_id}")

batch = client.messages.batches.retrieve(batch_id)
print(f"Status: {batch.processing_status}")

if batch.processing_status != "ended":
    counts = batch.request_counts
    print(f"Not done yet. Processing: {counts.processing}, check back later.")
else:
    results_file = SCRIPT_DIR / "results" / f"{batch_id}.jsonl"
    lines = []
    for result in client.messages.batches.results(batch_id):
        lines.append(json.dumps(result.model_dump()))  # model_dump() converts the SDK's Pydantic object to a plain dict
    results_file.write_text("\n".join(lines))
    batch_id_file.unlink()
    counts = batch.request_counts
    print(f"Done! Results saved to {results_file}")
    print(f"Succeeded: {counts.succeeded}, Errored: {counts.errored}, Expired: {counts.expired}")
