import json
from pathlib import Path
from google import genai

SCRIPT_DIR = Path(__file__).parent
client = genai.Client()

batch_name_file = SCRIPT_DIR / "batch_id_google.txt"
custom_ids_file = SCRIPT_DIR / "batch_custom_ids_google.json"

batch_name = batch_name_file.read_text().strip()
custom_ids = json.loads(custom_ids_file.read_text())
print(f"Checking batch: {batch_name}")

TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

batch = client.batches.get(name=batch_name)
print(f"Status: {batch.state.name}")

if batch.state.name not in TERMINAL_STATES:
    print("Not done yet. Check back later.")
elif batch.state.name != "JOB_STATE_SUCCEEDED":
    print(f"Batch ended with state: {batch.state}")
    if batch.error:
        print(f"Error: {batch.error}")
else:
    safe_name = batch_name.replace("/", "_")  # batch names contain slashes (e.g. "batches/abc123"), sanitise for filename
    results_file = SCRIPT_DIR / "results" / f"{safe_name}.jsonl"
    lines = []
    for i, resp in enumerate(batch.dest.inlined_responses):  # results are inlined in the job object, not a separate download
        custom_id = custom_ids[i] if i < len(custom_ids) else f"unknown_{i}"
        if resp.error:
            record = {"custom_id": custom_id, "error": str(resp.error)}
        else:
            record = {"custom_id": custom_id, "response": {"text": resp.response.text}}
        lines.append(json.dumps(record))
    results_file.write_text("\n".join(lines), encoding="utf-8")
    batch_name_file.unlink()
    custom_ids_file.unlink()
    print(f"Done! Results saved to {results_file}")
    print(f"Total: {len(lines)} results, Errors: {sum(1 for r in lines if 'error' in json.loads(r))}")
