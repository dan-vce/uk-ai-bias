# Bias in LLMs for UK Public Sector Housing Allocation

Undergraduate dissertation investigating allocational bias in large language models applied to UK council housing scenarios. The study tests whether LLMs assign different priority scores to applicants based on protected demographic characteristics.

This repository may be used to analyse results, or rerun experiments e.g. testing against a new model.

## Repository Structure

```
prompt-generation/         Prompt datasets and generation scripts
  allocational/            Housing allocation scenarios + prompts
    scenarios.json
    baseline_prompts.json
    intersectional_prompts.json
    names_prompts.json
    generate_allocational_prompts.py

{anthropic,openai,google,together}/
  jsonconverter_*.py       Convert prompts JSON → platform batch JSONL
  submit_batch_*.py        Upload JSONL and start batch job
  retrieve_batch_*.py      Poll for completion and download results
  ready_to_submit/         Prepared JSONL files
  results/                 Raw and parsed results + per-model analysis

parse_*.py                 Parse raw JSONL results into structured JSON
analysis.py                Cross-model analysis and figure generation
tests.py                   DeepEval fairness tests

figures/                   Output charts (MAE and mean signed difference)
analysis_summary.xlsx      Summary statistics across all models
results_by_scenario_region.csv
```

## Pipeline

### Dataset Generation

<img width="1203" height="353" alt="project flow1" src="https://github.com/user-attachments/assets/3761bc5e-721b-418d-af07-99fdc7375b4c" />

### Dataset Processing

<img width="687" height="353" alt="project flow2" src="https://github.com/user-attachments/assets/b91c74e0-91a2-4c40-b4e1-014b95009446" />


```
1. Generate prompts
   python prompt-generation/allocational/generate_allocational_prompts.py

2. Convert to batch format (per platform)
   python anthropic/jsonconverter_anthropic.py <prompts.json> <system_prompt.txt>

3. Submit batch
   python anthropic/submit_batch_anthropic.py batch_input_anthropic.jsonl

4. Retrieve results (when batch completes)
   python anthropic/retrieve_batch_anthropic.py

5. Parse results
   python parse_baseline_results.py
   python parse_intersectional_results.py
   python parse_names_results.py

6. Analyse
   python analysis.py
```
## Models Tested

| Provider   | Model                        | Notes                     |
|------------|------------------------------|---------------------------|
| Anthropic  | `claude-haiku-4-5-20251001`  | Extended thinking enabled |
| OpenAI     | `gpt-5.4-mini`               | Low-effort reasoning      |
| Google     | `gemini-3-flash-preview`     | Low thinking level        |
| Together   | `openai/gpt-oss-120b`        | Open-source baseline      |

## Prompt Design

Three prompt variants test different channels of demographic signal:

- **Baseline** — no demographic information (control)
- **Intersectional** — explicit demographic attributes in the prompt
- **Names** — implicit demographics via culturally associated first names

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Set API keys as environment variables:

```
ANTHROPIC_API_KEY
OPENAI_API_KEY
GOOGLE_API_KEY
TOGETHER_API_KEY
```
