# PEARL: Predictor-Agnostic Experience Retrieval for Popularity Forecasting and Recommendation

Code for the experiments in our paper (details withheld for review).
The pipeline is run on three datasets: **Douban** (movies) and **Amazon** (Baby, Home & Kitchen).

> Raw data, intermediate files, and API outputs are not included. See "Data" below.

## Framework overview

PEARL has three components (see the figure above). The code for each is listed in the [Pipeline](#pipeline) table below.

1. **Experience Construction.** Past observations are turned into a reusable experience base.
   - *Data Construction*: item popularity sequences are cut into sliding windows, and an LLM writes the hindsight factors that led to each observed outcome.
   - *Grouping and Profiling*: training items are assigned to outcome groups by observed popularity level (high, medium, zero) and direction (increase, decrease, stable). Statistical analyses and correlation studies then profile each group.
2. **Experience Retrieval.** For a query item, retrieval is done in two steps.
   - *Outcome Group Selection*: a data-analyst summary and an LLM feature selector (which picks the top features from the candidate features) are used to score the groups and choose the single most similar group.
   - *Within-group Case Retrieval*: inside that group, items are prefiltered by embedding similarity and reranked by semantic-temporal similarity. The top-K items are formatted as the retrieved context.
3. **Predictor-Agnostic Forecasting.** The retrieved context is put into the prediction prompt of any predictor LLM (GPT-4o-mini, Llama-3, Qwen2, DeepSeek). No predictor is trained or fine-tuned; the folder only contains the prediction prompts and the scripts that call each predictor. The predicted popularity scores are sorted to give the next month's top recommendations.

Everything else in this repository (evaluation, baselines, ablations, the complementary study, retrieval-quality analysis and API utilities) lies outside the framework itself.

## Setup

```bash
pip install -r requirements.txt   # Python 3.12
export OPENAI_API_KEY=...      # explanation generation, GPT-4o-mini prediction
export DEEPSEEK_API_KEY=...    # DeepSeek prediction
# optional: export EMBEDDING_MODEL=/path/to/local/model   (default: microsoft/harrier-oss-v1-270m)
```

A GPU is needed for embedding generation and for serving Llama-3 / Qwen2 with vLLM.

## Pipeline

The folders are numbered in execution order, so one framework component can span several folders. Use `--help` on any script
for its arguments (script numbers `sN_...` are not contiguous).

| Framework component | Step | Code |
|---|---|---|
| **1. Experience Construction** | Data Construction | `01_preprocessing/` (train/test sets per dataset); `02_explanation_and_embedding/01_explanation_generation/` (LLM hindsight factors, prompts in `system_prompts/`); `02_.../02_embedding_generation/` (item embeddings) |
| | Grouping and Profiling | `03_clustering_and_indexing/`: `s5_save_eight_groups.py` (outcome groups and their profiles), `s6`-`s8` (correlation studies) |
| **2. Experience Retrieval** | Outcome Group Selection (top-1 most similar group) | `04_data_analyst_feature_selection/` (data-analyst summary, feature selector prompts and batches); `05_retrieval_and_formatting/s14_retrieve_top1_centers.py` |
| | Within-group Case Retrieval (and item formatting) | `03_clustering_and_indexing/s16_indexing.py` (scored member index); `05_retrieval_and_formatting/s17_...py` (top-K retrieval, prefilter + reranking, card formatting) |
| **3. Predictor-Agnostic Forecasting** | Prediction prompts for each predictor | `06_prediction_and_evaluation/`: `gpt4omini/`, `llama3/`, `qwen2/`, `deepseek/` |
| **Outside the framework** | Evaluation, baselines, analyses | `06_.../s23`, `s24`, `metric_measurement_*_utils.py` (evaluation), `top_pop/` (baseline), `trail_retrieval/` (TRAIL+Retrieval prompts); `07_retrieval_quality/`; `08_complementary_study/`; `supporting_api_scripts/` (OpenAI Batch API helpers) |

## Batch API workflow

The LLM steps that use the OpenAI Batch API all follow the same pattern: a prepare script writes a batch `.jsonl` plus a mapping CSV
keyed by `custom_id`; the batch is run with `supporting_api_scripts/`; the results are joined back to the mapping on `custom_id`
with `s13_merging_api_calls.py`. `s12_batch_pipeline.py` parses each reply as JSON into columns (and can `retry`/`cancel`);
`s12_batch_exp_pipeline.py` keeps the free-text reply as `explanatory_factors`. `submit` stores `<tag>_batch_meta.json` in the
current directory, so run `submit`, `status` and `download` from the same directory with the same `--tag`.

1. **Hindsight factors (labelled as explanatory_factors).** `02_.../01_explanation_generation/s4_prepare_batch_input.py` writes `<dataset>_full_data.jsonl` and
   `<dataset>_mapping_full_data.csv`. Run it with `s12_batch_exp_pipeline.py` (`submit`, `status`, `download`), then
   `s13_merging_api_calls.py --mapping <dataset>_mapping_full_data.csv --results <downloaded csv> --output <train csv with explanatory_factors>`.
2. **Feature selection.** `04_.../s11_feature_slx_prep_batch.py` writes `<tag>_batch.jsonl` and a feature-selection mapping CSV. Run it with
   `s12_batch_pipeline.py` (`submit`, `status`, `download`, `retry`; use `s111_batch_splitter.py` first if the file exceeds the
   batch size limit), then merge with `s13_merging_api_calls.py` to get `<dataset>_feature_selection_merged.csv`, which `s14` and `s17` read.
3. **Prediction.** `06_.../gpt4omini/s18_*` writes `pred_batch_<tag>.jsonl` and a `mapping_<tag>.csv`. Run the batch with
   `s12_batch_pipeline.py` (`submit`, `status`, `download` -- without `--mapping-csv`, which is only for feature selection); `download`
   writes `<tag>.csv` with `custom_id`, `predict_popularity_score` and `reasoning`. Malformed or missing outputs can be re-requested with
   `06_.../gpt4omini/s21_repair_gpt4o_mini.py`. The evaluation script `06_.../s23_run_model_evaluation_all_datasets.py` then joins the
   mapping with these predictions on `custom_id` itself.

## Ablation settings

There are two families of ablations: (A) the **retrieval mechanism** (table below), and (B) the **content of the retrieved items**
on top of the full G1E1 setup (section after the table).

(A) Rows of the retrieval-mechanism ablation table in the paper and how to reproduce each one. Retrieval settings are flags of
`05_retrieval_and_formatting/s17_retrieved_item_formatting_component_ablation.py`; feature-selector and random baselines are flags
of `s14_retrieve_top1_centers.py`. The retrieved context is then fed to the prediction scripts in `06_prediction_and_evaluation/`.

Defaults (the full framework, G1E1): `--group-routing scored --item-selection rerank --bucket-match on --alpha-embed 0.5
--beta-secondary 0.5 --k-values 5,10 --e-variants E1 --g-tag G1`. Every row below changes only the flags listed.
"+ factors" means `E1` (factors included); `E0` removes them.

| Paper row | What is ablated | How to run |
|---|---|---|
| No retrieval | the whole framework | the `no_retrieval` variant of the prediction scripts in `06_prediction_and_evaluation/` |
| Random retrieval + factors | cluster, pre-embed filter, semantic-temporal reranking | `s14 --random-group --random-seed N`, then `s17 --item-selection random` with that top-1 file |
| Semantic retrieval + factors | temporal reranking | `s17 --alpha-embed 1.0 --beta-secondary 0.0` |
| Temporal retrieval + factors | semantic reranking | `s17 --alpha-embed 0.0 --beta-secondary 1.0` |
| Semantic-temporal retrieval + factors | cluster | `s17 --group-routing none` |
| Full w random feature selection | LLM feature selection | `s14 --random-features --n-random-features 3 --random-seed N`, then `s17` with that top-1 file and its merged features file |
| Full w/o reranking | pre-embed filter + semantic-temporal reranking | `s17 --item-selection group_top` |
| Full w/o feature selection fixed ranking | fixed feature ranking | `s14 --no-feature-selector-active`, then `s17` with the inactive top-1 file and its features file |
| Full w/o feature selection fixed ranking + w/o reranking | both of the above | as above, plus `s17 --item-selection group_top` |
| Full w/o statistical guidance (G0E1) | statistical analysis | feature selector run with the G0 prompt (`04_.../feature_slx_system_prompts/g0e1_douban/`), `s14` active, then `s17 --g-tag G0 --e-variants E1` |
| Ours (G1E1) | nothing | defaults |

Example (full framework on one dataset; replace the placeholders with your own files):

```bash
python3 s17_retrieved_item_formatting_component_ablation.py --dataset baby \
  --query-top1-csv $TOP1_ACTIVE --query-features-csv $FEATURES_MERGED --query-embed-path $TEST_EMBED \
  --group-routing scored --item-selection rerank --e-variants E1 --k-values 5,10 \
  --alpha-embed 0.5 --beta-secondary 0.5 --g-tag G1 \
  --output-dir processed_results/baby/retrieved_context_g1e1
```

(B) Retrieved-item content ablations, run on the full G1E1 setup with one component removed from each retrieved item:

| Removed from the retrieved items | How to run |
|---|---|
| Trend | `s17 --ablate-component trend` |
| Item name (and release date) | `s17 --ablate-component item_name` |
| Relative change | `s17 --ablate-component relative_change` |
| Hindsight factors | `s17 --e-variants E0` (G1E0) |

Output files of the `--ablate-component` runs get a `_noTrend` / `_noItemName` / `_noRelChange` suffix.

## Complementary study (`08_complementary_study/`)

Tests whether the predicted popularity is a useful complement to personalized recommenders (HGN, LLM-ESR):

    s_new(u, i, T) = beta * s(u, i) + (1 - beta) * y_hat_F

1. `s22_avg_for_combined_ranking.py` averages the predicted popularity of 3 runs per item, min-max normalizes it, and ranks the items (non-personalized recommender; also produces `norm_rank10` / `norm_rank20`).
2. `s23_score_mixture_multidataset.py` min-max normalizes the personalized scores globally and mixes them with the popularity score for each `beta` (default grid `0,0.05,...,1.0`).
3. `s24_evaluate_hr_ndcg_2.py` computes HR / NDCG / Reference NDCG for every `(model, beta)`; `s25_reorganize_personalized_results.py` formats the result tables.

The per-user scores of HGN and LLM-ESR are produced by their own training code (not included); the scripts expect the exported `user_item_scores.csv.gz` and `test_interactions.csv` files. For Baby and Douban the models' dense NPZ export is first converted to that long format with `convert_npz_to_long.py`. `metrics.py` implements HR / NDCG / Reference NDCG / Recall / Precision used by `s24`.

## TRAIL + Retrieval

To test whether retrieval also helps an existing method, we run the TRAIL baseline with the only change being its
system prompt: it is extended to describe the `<SimilarPastItems>` block (retrieved similar past items, from the
retrieval pipeline in `05_retrieval_and_formatting/`). The TRAIL code itself is **not** modified and is not included; use the authors'
public implementation (https://github.com/Revive-dontwanttocode/TRAIL, `master` branch) and replace its system prompt with:

- `06_prediction_and_evaluation/trail_retrieval/prompts/douban.txt` (Douban movies)
- `06_prediction_and_evaluation/trail_retrieval/prompts/amazon.txt` (Amazon Home & Kitchen)

Evaluation uses the same scripts as the other experiments (`06_prediction_and_evaluation/s23_...`), which apply TRAIL's NDCG formula.

## Models used

| Step | Model |
|---|---|
| Explanatory-factor generation (`02_.../01_explanation_generation/s4_prepare_batch_input.py`) | `gpt-5.6-luna` (OpenAI API, `reasoning_effort="none"`), T=0.1 |
| Feature selection (`04_data_analyst_feature_selection/s11_feature_slx_prep_batch.py`) | `gpt-5.6-luna` (same settings), T = 0 |
| Embeddings (`02_.../02_embedding_generation/`) | `microsoft/harrier-oss-v1-270m` |
| Popularity prediction | `gpt-4o-mini-2024-07-18` (default), plus Llama-3-8B, Qwen2-7B (vLLM) and DeepSeek for comparison, T = 0.1 except DeepSeek 0.05 |

## Data

Scripts read and write CSV/NPY files using relative paths (e.g. `raw_dataset/`, `dataset/`,
`processed_results/`). Place the datasets there, or edit the paths at the top of each script.
Douban and Amazon datasets must be obtained from their original sources.

## Artifacts

`artifacts/group_profiles/` holds the intermediate outputs of the grouping and profiling step for each dataset (Douban, Amazon Baby,
Amazon Home & Kitchen), so the groups and statistics the retrieval and prompts rely on can be inspected without the raw data.
The folder names start with the date the run was made. They contain derived summaries only, no raw item text or reviews.

| File | Contents | Produced by |
|---|---|---|
| `group_profiles.json` | statistics and narration of each outcome group and history-depth bucket | `03_.../s5_save_eight_groups.py` |
| `group_narrations_for_prompt.json` / `.txt` | the per-group narrations that are inserted into the prompts | `s5_save_eight_groups.py` |
| `group_mean_embeds.npz` | mean embedding of each group and bucket | `s5_save_eight_groups.py` |
| `group_doc_ids.npz` | internal item indices (not original item ids) belonging to each group and bucket | `s5_save_eight_groups.py` |
| `*_group_scored_members.json` | members of each group scored against their group center, sorted | `03_.../s16_indexing.py` |
| `stats.txt`, `notes.txt` | group and bucket counts | `s5_save_eight_groups.py` |
| `corr_data/` | age-stratified correlation outputs, bucket composition, item-age correlation study, pairwise separability | `03_.../s6`, `s7`, `s8` |

## Notes

- Douban timestamps are interpreted in UTC+8; keep this fixed to reproduce the reported numbers.
- Scripts in `05_retrieval_and_formatting/` and `07_retrieval_quality/` import shared modules (`refined_retrieval_utils`, `dataset_config`, ...) from `03_clustering_and_indexing/` via `sys.path`; Amazon prediction scripts import `s18_prepare_for_prediction_amazon` from `gpt4omini/`.
- Evaluation metrics live in `06_prediction_and_evaluation/metric_measurement_*_utils.py`.
