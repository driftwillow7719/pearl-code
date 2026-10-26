# PEARL: Predictor-Agnostic Experience Retrieval for Popularity Forecasting and Recommendation

Code for the experiments in our paper (details withheld for review), run on Douban (movies) and Amazon (Baby, Home & Kitchen).

> Raw data, intermediate files, and API outputs are not included. See "Data" below.

## Setup

```bash
pip install -r requirements.txt   # Python 3.12
export OPENAI_API_KEY=...         # explanatory factors, feature selection, GPT-4o-mini prediction
export DEEPSEEK_API_KEY=...       # DeepSeek prediction
```

A GPU is needed for embedding generation and for serving Llama-3 / Qwen2 with vLLM. Use `--help` on any script for its arguments.

## Code overview

Folders are numbered in execution order, so one component of the framework can span several folders.

| Component | Step | Code |
|---|---|---|
| **1. Experience Construction** | Data Construction | `01_preprocessing/`, `02_.../00_synthetic_data_generation/`, `02_.../01_explanation_generation/`, `02_.../02_embedding_generation/` |
| | Grouping and Profiling | `03_clustering_and_indexing/` (`s5_save_eight_groups.py`, correlation studies `s6`-`s8`) |
| **2. Experience Retrieval** | Outcome Group Selection | `04_data_analyst_feature_selection/`, `05_retrieval_and_formatting/s14_retrieve_top1_centers.py` |
| | Within-group Case Retrieval | `03_.../s16_indexing.py`, `05_retrieval_and_formatting/s17_...py` |
| **3. Predictor-Agnostic Forecasting** | Prediction prompts per predictor | `06_prediction_and_evaluation/system_prompts/`|
| Outside the framework | Evaluation, baselines, analyses | `06_.../s23`, `s24`, `metric_measurement_*_utils.py`, `top_pop/`, `trail_retrieval/`; `07_retrieval_quality/`; `08_complementary_study/` |
| Utilities | OpenAI Batch API helpers | `supporting_api_scripts/` |

### Batch API steps

Explanatory factors (`s4`), feature selection (`s11`) and GPT-4o-mini prediction (`s18`) write a batch `.jsonl` and a mapping CSV keyed by
`custom_id`. Run the batch with `supporting_api_scripts/` (`s12_batch_pipeline.py` or `s12_batch_exp_pipeline.py`: `submit`, `status`, `download`;
run them from the same directory with the same `--tag`), then join the results to the mapping with `s13_merging_api_calls.py`.
The prediction results are joined to their mapping by the evaluation script `s23` itself.

## Ablations

All runs use `05_.../s17_retrieved_item_formatting_component_ablation.py` (and `s14_retrieve_top1_centers.py` where noted). Defaults (full framework, G1E1):
`--group-routing scored --item-selection rerank --alpha-embed 0.5 --beta-secondary 0.5 --k-values 5,10 --e-variants E1 --g-tag G1`.

**Retrieval mechanism** (rows of the ablation table in the paper; "+ factors" = `E1`):

| Paper row | What is ablated | How to run |
|---|---|---|
| No retrieval | the whole framework | the `no_retrieval` variant of the prediction scripts |
| Random retrieval + factors | cluster, pre-embed filter, semantic-temporal reranking | `s14 --random-group --random-seed N`, then `s17 --item-selection random` |
| Semantic retrieval + factors | temporal reranking | `s17 --alpha-embed 1.0 --beta-secondary 0.0` |
| Temporal retrieval + factors | semantic reranking | `s17 --alpha-embed 0.0 --beta-secondary 1.0` |
| Semantic-temporal retrieval + factors | cluster | `s17 --group-routing none` |
| Full w random feature selection | LLM feature selection | `s14 --random-features --n-random-features 3 --random-seed N`, then `s17` |
| Full w/o reranking | pre-embed filter + semantic-temporal reranking | `s17 --item-selection group_top` |
| Full w/o feature selection fixed ranking | fixed feature ranking | `s14 --no-feature-selector-active`, then `s17` |
| Full w/o feature selection fixed ranking + w/o reranking | both of the above | as above, plus `s17 --item-selection group_top` |
| Full w/o statistical guidance (G0E1) | statistical analysis | G0 prompt (`04_.../feature_slx_system_prompts/g0e1_douban/`), then `s17 --g-tag G0 --e-variants E1` |
| Ours (G1E1) | nothing | defaults |

**Retrieved-item content** (one component removed from each retrieved item, on top of G1E1):
`s17 --ablate-component trend|item_name|relative_change`, or `s17 --e-variants E0` to remove the explanatory factors.

## Complementary study (`08_complementary_study/`)

Combines a personalized recommender's score with our predicted popularity, `s_new = beta * s(u, i) + (1 - beta) * y_hat_F`:
`s22` averages the predicted popularity over 3 runs, min-max normalizes and ranks it; `s23` mixes it with the HGN / LLM-ESR scores for each `beta`;
`s24` computes HR / NDCG (using `metrics.py`); `s25` formats the tables. The HGN and LLM-ESR scores come from their own code (not included);
for Baby and Douban their NPZ export is converted with `convert_npz_to_long.py`.

## TRAIL + PEARL

We run the TRAIL baseline with only its system prompt changed, extended to describe the retrieved `<SimilarPastItems>` block. The TRAIL code
is not modified or included; use the authors' implementation (https://github.com/Revive-dontwanttocode/TRAIL, `master` branch) with the prompts in
`06_prediction_and_evaluation/trail_retrieval/prompts/` (`douban.txt`, `amazon.txt`).

## Models used

| Step | Model |
|---|---|
| Explanatory-factor generation (`02_.../01_explanation_generation/s4_prepare_batch_input.py`) | `gpt-5.6-luna` (OpenAI API, `reasoning_effort="none"`), T=0.1 |
| Feature selection (`04_data_analyst_feature_selection/s11_feature_slx_prep_batch.py`) | `gpt-5.6-luna` (same settings), T = 0 |
| Embeddings (`02_.../02_embedding_generation/`) | `microsoft/harrier-oss-v1-270m` |
| Popularity prediction | `gpt-4o-mini-2024-07-18` (default), plus Llama-3-8B-Instruct, Qwen2-7B-Instruct (served with vLLM; `--model` is the served model name) and DeepSeek-V4.1-Flash for comparison, T = 0.1 except DeepSeek-V4.1-Flash 0.05 |

## Data and artifacts

Scripts read and write CSV/NPY files using relative paths (e.g. `raw_dataset/`, `dataset/`, `processed_results/`). Place the datasets there or edit the
paths at the top of each script. Douban and Amazon must be obtained from their original sources.

`artifacts/group_profiles/` has the derived outputs of the grouping and profiling step for each dataset (Douban, Baby, Home & Kitchen); they are
summaries only, no raw item text or reviews:

- `group_profiles.json`, `group_narrations_for_prompt.*`, `group_mean_embeds.npz`, `group_doc_ids.npz` (internal item indices): produced by `03_.../s5_save_eight_groups.py`
- `*_group_scored_members.json`: members scored against their group center, from `03_.../s16_indexing.py`
- `corr_data/`: correlation studies from `s6`-`s8`

## Notes

- Douban timestamps are interpreted in UTC+8; keep this fixed to reproduce the reported numbers.
- Scripts in `05_retrieval_and_formatting/` and `07_retrieval_quality/` import shared modules from `03_clustering_and_indexing/` via `sys.path`.
