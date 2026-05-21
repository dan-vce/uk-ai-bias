from pathlib import Path
from together import Together

SCRIPT_DIR = Path(__file__).parent
client = Together()

batch_id_file = SCRIPT_DIR / "batch_id_together.txt"
batch_id = batch_id_file.read_text().strip()
print(f"Checking batch: {batch_id}")

# .retrieve() returns the batch directly, unlike .create() which wraps it in .job
batch = client.batches.retrieve(batch_id)
print(f"Status: {batch.status} | Progress: {batch.progress:.0f}%")

if batch.status != "COMPLETED":
    if batch.status in ("FAILED", "EXPIRED", "CANCELLED"):
        print(f"Batch ended with status: {batch.status}")
        if batch.error:
            print(f"Error: {batch.error}")
    else:
        print("Not done yet. Check back later.")
else:
    results_file = SCRIPT_DIR / "results" / f"{batch_id}.jsonl"
    # stream to avoid loading the full results file into memory
    with client.files.with_streaming_response.content(id=batch.output_file_id) as response:
        with results_file.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    batch_id_file.unlink()
    print(f"Done! Results saved to {results_file}")

    if batch.error_file_id:  # Together separates failed rows into a second file
        errors_file = SCRIPT_DIR / "results" / f"{batch_id}_errors.jsonl"
        with client.files.with_streaming_response.content(id=batch.error_file_id) as response:
            with errors_file.open("wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
        print(f"Errors saved to {errors_file}")
