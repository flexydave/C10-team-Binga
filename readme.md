# African Folktales SLM Challenge — Document Retrieval

## Project Overview

This repository contains a retrieval-based submission method for the African Folktales SLM Challenge. For each test prompt, it retrieves one corpus document and submits that document's story text. This README covers retrieval only; it does not describe fine-tuning or LoRA. The dataset metadata defines the competition metric as mean Levenshtein distance: the average edit distance between prediction and reference story, where lower is better.

## Dataset

The retrieval notebook uses the Kaggle competition input at `/kaggle/input/competitions/african-folktales-slm-challenge`.

- `data/documents.csv` has 24 unique `document_id` rows and eight columns: `document_id`, `title`, `theme`, `culture_region`, `text`, `origin`, `source_url`, and `license`. All 24 documents are indexed. The corpus has six themes with four documents each; `source_url` is blank for every row and is not used.
- `data/train_prompts.csv` has 38 prompt/story pairs: `prompt`, `theme`, `culture_region`, `document_id`, `reference_story`, and `PromptId`. Its pairs cover all 24 document IDs, but the notebook only loads and displays this file; it does not fit, tune, or evaluate retrieval with it.
- `data/test_prompts.csv` has 10 rows with `PromptId`, `prompt`, `theme`, and `culture_region`. These rows form the retrieval queries.

The notebook also loads `sample_submission.csv` and `baseline_submission.csv` from the Kaggle input, but neither is used after loading. Their required submission schema is `PromptId,Story`; the two files are not committed under this repository's `data/` directory. `data/dataset-metadata.json` records the same dataset schemas and counts.

## Training / Retrieval Pipeline

### Data Collection

`scripts/document-retrieval.ipynb` reads the five competition CSV files. The retrieval corpus is every row of `documents.csv`; it is not filtered or merged with the training pairs.

### Preprocessing

For each document, the notebook concatenates non-null `title`, `text`, `theme`, and `culture_region`. For each test query, it concatenates non-null `prompt`, `theme`, and `culture_region`. `clean_text` converts text to lowercase, replaces every character outside `[a-z0-9\s]` with a space, collapses repeated whitespace, and strips ends. It does not remove duplicates or tokenize separately.

### Retrieval Method

`TfidfVectorizer` is fitted on the 24 cleaned document strings. The notebook transforms the 10 cleaned test strings, computes `cosine_similarity(test_vectors, document_vectors)`, and uses `numpy.argmax` to choose one highest-scoring document per prompt. It prints the top three documents for inspection, but submission uses only rank one. The final `Story` is the selected document's original `text`, not its cleaned retrieval text.

### Hyperparameters / Design Choices

- TF–IDF: `ngram_range=(1, 2)`, `min_df=1`, `sublinear_tf=True`.
- Selection: cosine similarity followed by a single top-1 `argmax`; no threshold or fallback is implemented.
- No hyperparameter search is documented.

## Evaluation

The notebook has no holdout, leave-one-out, training reconstruction, mean-Levenshtein, baseline-comparison, or Kaggle-score calculation. `reference_story` is never used by retrieval. Its recorded output shows a 24×1,827 document matrix, a 10×24 similarity matrix, and a printed top-three ranking for each test prompt.

For competition scoring, mean Levenshtein distance is the average edit distance between predicted and reference stories; lower is better. The submission loop visits each test row in file order and carries its `PromptId` into one output row with columns `PromptId` and `Story`. No explicit schema, missing-value, row-count, or order assertion is implemented, so check these against `sample_submission.csv` before upload.

## Reproduction

No Python version or `requirements.txt` is committed. Use a Kaggle Python notebook with the `african-folktales-slm-challenge` competition input attached; the notebook imports NumPy, pandas, scikit-learn, `re`, `os`, and `kagglehub` (the last is not used by retrieval).

1. Open `scripts/document-retrieval.ipynb` in Kaggle.
2. Correct the first cell before running: its pandas import and following `from sklearn...` statement are joined on one line, which is invalid Python. Separate them with a newline.
3. Run cells in order: load data; inspect it; clean and build the document index; transform test queries and score; inspect matches/top three; build the submission.
4. The final cell writes `/kaggle/working/submission_retrieval.csv`. Before submitting, compare its 10 rows, `PromptId` order, `PromptId,Story` columns, and null values with Kaggle's `sample_submission.csv`.

## Repository Structure

```text
.
├── data/
│   ├── documents.csv
│   ├── train_prompts.csv
│   ├── test_prompts.csv
│   └── dataset-metadata.json
├── scripts/
│   └── document-retrieval.ipynb
└── README.md
```

## Limitations

This is extractive retrieval, so it cannot compose a new folktale. Results depend on the 24-document corpus and on lexical TF–IDF overlap; semantically similar prompts can select an unsuitable story. No committed validation quantifies retrieval quality against the competition metric.

## Appendix: Contributors and Mentors

### Contributors / Team Members

- DAVID ATTAH, BRIGHT FRANCIS, ORANGUN FOLARANMI

### Mentors

- PATRICK OWOR

## References

- [Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks)
- [scikit-learn: `TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn: cosine similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
- [pandas documentation](https://pandas.pydata.org/docs/)
