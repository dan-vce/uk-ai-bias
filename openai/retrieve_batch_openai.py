from pathlib import Path
from openai import OpenAI

SCRIPT_DIR = Path(__file__).parent
client = OpenAI()

batch_id_file = SCRIPT_DIR / "batch_id_openai.txt"
batch_id = batch_id_file.read_text().strip()
print(f"Checking batch: {batch_id}")

batch = client.batches.retrieve(batch_id)
print(f"Status: {batch.status}")

if batch.status != "completed":
    if batch.status in ["failed", "cancelled", "expired"]:
        print(f"Batch ended with status: {batch.status}")
    else:
        print(f"Not done yet. Check back later.")
else:
    results_file = SCRIPT_DIR / "results" / f"{batch_id}.jsonl"
    content = client.files.content(batch.output_file_id).text
    results_file.write_text(content)
    batch_id_file.unlink()
    print(f"Done! Results saved to {results_file}")
