# 🎵 MusicRecoSystem

**Music artist recommendation using Machine Learning and Deep Learning**

MusicRecoSystem explores how listening histories can be used to recommend **10 artists a user has not listened to before**. The project covers the full experimental workflow: collecting real-world data, preparing user–artist interactions, training recommendation models, and evaluating ranked suggestions.

## Tech Stack

**Python · Pandas · NumPy · Requests · Scikit-learn · XGBoost · PyTorch · Jupyter Notebook · JSON · Parquet · Git/GitHub**


## Data Collection & Preparation

Listening histories were collected through the **ListenBrainz API** using Python and Requests. Alongside collecting individual users' histories, **targeted artist-based sampling** was explored to find listeners with overlapping musical interests.

Raw JSON responses were cleaned and transformed with **Pandas and NumPy**, then stored as **Parquet** files. Duplicate submissions (same user, artist credit, track, and timestamp after text normalization) are removed, while genuine repeated listens at different times are retained as a signal of listening preference.

Artist identities are based on **MusicBrainz IDs (MBIDs)** rather than raw display strings. Collaborations are expanded to one row per credited artist, so an interaction with “Steve Lacy, SZA” contributes to the same Steve Lacy and SZA identities as their solo tracks. `artist_uid` is the stable modeling key, `artist_mbid` stores the MusicBrainz identifier, `artist` is the canonical display name, and `artist_credit` preserves the original full credit. When a raw listen has no MBID, the pipeline first tries to infer its artists from unambiguous aliases already present in the dataset. Unresolved cases are never discarded: they receive an explicit `text:` fallback identity with Unicode-normalized capitalization, spacing, accents, apostrophes, and dashes. Meaningful accents are preserved to avoid merging distinct non-Western artist names.

The processed data is then used to build user–artist interaction matrices for modeling.

Before modeling, unusable user profiles are filtered with explicit, configurable rules: at least 15 distinct artists, no single artist appearing in more than 50% of listening events, and at least 24 hours between the first and last listen. The complete decision trail is saved to `data/processed/user_quality.parquet`, including metrics, boolean flags, and exclusion reasons for every user.

Every cleaning run also creates `data/processed/cleaning_report.json` and an immutable snapshot under `data/processed/versions/`. The deterministic `dataset_version` depends on the raw-data fingerprint, pipeline version, and filtering thresholds; it is also embedded in every processed row. The journal records the row and user counts at every stage, exact removal reasons, configuration, file hashes, and output paths so notebook results can always be tied to one dataset version.

## Models Explored

| Model | Approach | Description |
|---|---|---|
| **K-Means** | Unsupervised learning | Groups listeners by their artist preferences and recommends popular artists within a cluster. |
| **XGBoost** | Supervised learning | Learns to score user–artist interactions using positive observations and sampled negatives. |
| **Neural Collaborative Filtering (PyTorch)** | Deep learning | Learns user and artist embeddings with an MLP to predict interaction scores. |
| **Popularity baseline** | Benchmark | Recommends globally popular unseen artists as a simple comparison. |

The PyTorch model uses **ReLU activations, binary cross-entropy with logits, Adam optimization, and negative sampling**. Experiments varied embedding size and weight decay.

## Project Structure

| File | Responsibility |
|---|---|
| `src/clean.py` | Cleaning, artist identity resolution, user-quality filtering, versioning, and audit logs. |
| `src/recommend.py` | Shared evaluation utilities and the reusable K-Means recommender. |
| `src/recommend_xgboost.py` | XGBoost feature engineering, supervised sampling, tuning, final training, and recommendations. |
| `src/recommend_deep_learning.py` | PyTorch NCF architecture, negative sampling, tuning, final training, and recommendations. |
| `notebook/Kmeans.ipynb` | Minimal K-Means usage example. |
| `notebook/XGBoost.ipynb` | Minimal XGBoost usage example. |
| `notebook/DeepLearning.ipynb` | Minimal NCF usage example. |

## Experimental Results

Models were evaluated using **Recall@10** (coverage of held-out relevant artists) and **NDCG@10** (ranking quality, rewarding relevant artists placed higher).

### K-Means experiment

The reusable pipeline in `src/recommend.py` performs a chronological 80/10/10 split, selects `k` using validation NDCG@10, and then automatically retrains the final model on train + validation. With dataset `clean-v1.0.0-19b966482277`, `k=4` was selected from the candidate grid used by the notebook.

| Split | k | Evaluable users | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Validation | 4 | 67 | 0.1530 | 0.1196 |
| Held-out test | 4 | 66 | 0.1313 | 0.1075 |

### XGBoost experiment

The pipeline compares three configurations using validation NDCG@10. The **shallow** configuration (150 estimators, depth 2, learning rate 0.05) was selected and retrained on train + validation.

| Split | Configuration | Evaluable users | Recall@10 | NDCG@10 |
|---|---|---:|---:|---:|
| Validation | shallow | 67 | 0.1649 | 0.1111 |
| Held-out test | shallow | 66 | 0.1163 | 0.0925 |

### PyTorch NCF experiment

Three configurations were trained for five epochs. The validation selected an 8-dimensional embedding with weight decay `1e-4`, which was then retrained on train + validation.

| Split | Configuration | Evaluable users | Recall@10 | NDCG@10 |
|---|---|---:|---:|---:|
| Validation | embedding 8 + regularization | 67 | 0.0557 | 0.0646 |
| Held-out test | embedding 8 + regularization | 66 | 0.0331 | 0.0250 |

All three model families now use the same dataset version, artist identities, chronological split, and evaluation definitions, so their held-out results are directly comparable.



## Current Limitations

- **Sparse interactions:** Many users share few artists, limiting collaborative signals.
- **Limited data:** More varied and overlapping listening histories may help complex models generalize.
- **Cold start:** Models based on learned user IDs cannot directly serve a completely new user.
- **Catalog coverage:** Artists absent from the training catalog cannot be recommended by the current models.
- **Implicit feedback:** An unobserved listen does not necessarily indicate dislike; negative sampling introduces noise.
- **Model performance:** Popularity remains a strong baseline, outperforming the tested NCF models in their evaluation.
- **Neural-model performance:** The current NCF model needs more data and tuning before it can compete with the simpler approaches.

## Next Steps

The next priorities are a same-protocol popularity baseline, model persistence, support for users who provide favorite artists, expanded listening data, and further tuning. Apple Music history integration may be explored later.

---

**Project status:** Versioned preprocessing and reusable K-Means, XGBoost, and PyTorch recommendation pipelines are operational.
# -Music-recommendation-system
