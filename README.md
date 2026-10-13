# 🎵 MusicRecoSystem

**Music artist recommendation using Machine Learning and Deep Learning**

MusicRecoSystem explores how listening histories can be used to recommend **10 artists a user has not listened to before**. The project covers the full experimental workflow: collecting real-world data, preparing user–artist interactions, training recommendation models, and evaluating ranked suggestions.

## Tech Stack

**Python · Pandas · NumPy · Requests · Scikit-learn · XGBoost · PyTorch · Jupyter Notebook · JSON · Parquet · Git/GitHub**


## Data Collection & Preparation

Listening histories were collected through the **ListenBrainz API** using Python and Requests. Alongside collecting individual users' histories, **targeted artist-based sampling** was explored to find listeners with overlapping musical interests.

Raw JSON responses were cleaned and transformed with **Pandas and NumPy**, then stored as **Parquet** files. Repeated listens were retained as a signal of listening preference. The processed data was used to build user–artist interaction matrices for modeling.

## Models Explored

| Model | Approach | Description |
|---|---|---|
| **K-Means** | Unsupervised learning | Groups listeners by their artist preferences and recommends popular artists within a cluster. |
| **XGBoost** | Supervised learning | Learns to score user–artist interactions using positive observations and sampled negatives. |
| **Neural Collaborative Filtering (PyTorch)** | Deep learning | Learns user and artist embeddings with an MLP to predict interaction scores. |
| **Popularity baseline** | Benchmark | Recommends globally popular unseen artists as a simple comparison. |

The PyTorch model uses **ReLU activations, binary cross-entropy with logits, Adam optimization, and negative sampling**. Experiments varied embedding size and weight decay.

## Experimental Results

Models were evaluated using **Recall@10** (coverage of held-out relevant artists) and **NDCG@10** (ranking quality, rewarding relevant artists placed higher).

### K-Means experiment

Five clusters were selected using validation NDCG@10.

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| Global popularity | 0.1462 | 0.1018 |
| **K-Means (k=5)** | **0.1719** | **0.1182** |

### XGBoost experiment

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| Popularity | 0.0870 | **0.0658** |
| K-Means | **0.0872** | 0.0316 |
| XGBoost | 0.0749 | 0.0619 |

### PyTorch NCF experiment

Evaluated on **69 eligible users** with a chronological train/validation split. Each configuration was trained for **5 epochs**.

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| NCF — 8-dimensional embeddings | 0.1484 | 0.0783 |
| NCF — 8 dimensions + weight decay (seed 42) | 0.1244 | 0.0935 |
| **Popularity baseline** | **0.1862** | **0.1209** |

**Note:** These are results from **different experiments and evaluation setups**, not one shared leaderboard. Comparisons are meaningful primarily *within* each experiment. Some early NCF runs also used different random initializations, so their differences are preliminary.



## Current Limitations

- **Sparse interactions:** Many users share few artists, limiting collaborative signals.
- **Limited data:** More varied and overlapping listening histories may help complex models generalize.
- **Cold start:** Models based on learned user IDs cannot directly serve a completely new user.
- **Catalog coverage:** Artists absent from the training catalog cannot be recommended by the current models.
- **Implicit feedback:** An unobserved listen does not necessarily indicate dislike; negative sampling introduces noise.
- **Model performance:** Popularity remains a strong baseline, outperforming the tested NCF models in their evaluation.
- **Evaluation consistency:** A common data split and protocol are needed for rigorous cross-model comparisons.

## Next Steps

The next priority is to **turn the experiments into a reusable recommendation pipeline** that returns 10 unseen artists. Planned work includes support for users who provide favorite artists, more consistent evaluation, expanded listening data, and further model tuning. Apple Music history integration may be explored later.

---

**Project status:** Data collection, preprocessing, and initial K-Means, XGBoost, and PyTorch experiments completed. Recommendation pipeline consolidation is next.
# -Music-recommendation-system
