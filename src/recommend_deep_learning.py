"""Pipeline Neural Collaborative Filtering (PyTorch) pour les artistes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .recommend import (
    DEFAULT_ARTIST_COLUMN,
    DEFAULT_DISPLAY_COLUMN,
    chronological_split,
    evaluate_recommender,
    get_dataset_version,
)


DEFAULT_NCF_CONFIGS = (
    {"name": "embedding_8", "embedding_dim": 8, "weight_decay": 0.0},
    {"name": "embedding_16", "embedding_dim": 16, "weight_decay": 0.0},
    {
        "name": "embedding_8_regularized",
        "embedding_dim": 8,
        "weight_decay": 1e-4,
    },
)


class NeuralCollaborativeFiltering(nn.Module):
    """Réseau MLP utilisant des embeddings utilisateur et artiste."""

    def __init__(self, n_users: int, n_artists: int, embedding_dim: int = 8) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.artist_embedding = nn.Embedding(n_artists, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, user_ids: torch.Tensor, artist_ids: torch.Tensor) -> torch.Tensor:
        user_embedding = self.user_embedding(user_ids)
        artist_embedding = self.artist_embedding(artist_ids)
        combined = torch.cat([user_embedding, artist_embedding], dim=1)
        return self.mlp(combined).squeeze(-1)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NCFArtistRecommender:
    """Entraîne un NCF avec negative sampling et recommande des artistes inconnus."""

    def __init__(
        self,
        *,
        embedding_dim: int = 8,
        negative_ratio: int = 3,
        epochs: int = 5,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        random_state: int = 42,
        device: str | torch.device = "auto",
        user_col: str = "user",
        artist_col: str = DEFAULT_ARTIST_COLUMN,
        display_col: str = DEFAULT_DISPLAY_COLUMN,
        verbose: bool = False,
    ) -> None:
        if embedding_dim < 1 or negative_ratio < 1 or epochs < 1 or batch_size < 1:
            raise ValueError("Les dimensions, ratios, époques et batchs doivent être positifs")
        self.embedding_dim = embedding_dim
        self.negative_ratio = negative_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = _resolve_device(device)
        self.user_col = user_col
        self.artist_col = artist_col
        self.display_col = display_col
        self.verbose = verbose

    def _build_samples(self, positive_pairs: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        positives_by_user = (
            positive_pairs.groupby("user_id")["artist_id"].apply(set).to_dict()
        )
        all_artists = np.arange(len(self.artist_to_id_))
        negative_rows: list[tuple[int, int, float]] = []
        for user_id, known in positives_by_user.items():
            available = np.setdiff1d(all_artists, np.fromiter(known, dtype=int))
            sample_count = min(len(available), len(known) * self.negative_ratio)
            if sample_count:
                sampled = rng.choice(available, size=sample_count, replace=False)
                negative_rows.extend(
                    (int(user_id), int(artist_id), 0.0) for artist_id in sampled
                )

        positives = positive_pairs.copy()
        positives["label"] = 1.0
        negatives = pd.DataFrame(
            negative_rows, columns=["user_id", "artist_id", "label"]
        )
        samples = pd.concat([positives, negatives], ignore_index=True)
        return samples.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

    def fit(self, listens: pd.DataFrame) -> "NCFArtistRecommender":
        required = {self.user_col, self.artist_col}
        missing = required.difference(listens.columns)
        if missing:
            raise ValueError(f"Colonnes absentes : {', '.join(sorted(missing))}")
        if listens.empty:
            raise ValueError("Impossible d'entraîner NCF sur un DataFrame vide")

        _set_seed(self.random_state)
        users = sorted(listens[self.user_col].unique())
        artists = sorted(listens[self.artist_col].unique())
        self.user_to_id_ = {user: index for index, user in enumerate(users)}
        self.artist_to_id_ = {artist: index for index, artist in enumerate(artists)}
        self.id_to_artist_ = {index: artist for artist, index in self.artist_to_id_.items()}

        positive_pairs = listens[[self.user_col, self.artist_col]].drop_duplicates().copy()
        positive_pairs["user_id"] = positive_pairs[self.user_col].map(self.user_to_id_)
        positive_pairs["artist_id"] = positive_pairs[self.artist_col].map(self.artist_to_id_)
        positive_pairs = positive_pairs[["user_id", "artist_id"]]
        samples = self._build_samples(positive_pairs)

        dataset = TensorDataset(
            torch.tensor(samples["user_id"].to_numpy(), dtype=torch.long),
            torch.tensor(samples["artist_id"].to_numpy(), dtype=torch.long),
            torch.tensor(samples["label"].to_numpy(), dtype=torch.float32),
        )
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        self.model_ = NeuralCollaborativeFiltering(
            n_users=len(users),
            n_artists=len(artists),
            embedding_dim=self.embedding_dim,
        ).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.training_losses_: list[float] = []
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            total_loss = 0.0
            for user_ids, artist_ids, labels in loader:
                user_ids = user_ids.to(self.device)
                artist_ids = artist_ids.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model_(user_ids, artist_ids)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
            mean_loss = total_loss / len(dataset)
            self.training_losses_.append(mean_loss)
            if self.verbose:
                print(
                    f"  époque {epoch}/{self.epochs} — loss={mean_loss:.4f}",
                    flush=True,
                )

        self.known_artists_ = (
            listens.groupby(self.user_col)[self.artist_col].apply(set).to_dict()
        )
        self.catalog_ = set(artists)
        if self.display_col in listens.columns:
            self.artist_names_ = (
                listens[[self.artist_col, self.display_col]]
                .drop_duplicates(self.artist_col)
                .set_index(self.artist_col)[self.display_col]
                .to_dict()
            )
        else:
            self.artist_names_ = {}
        self.training_examples_ = len(samples)
        self.positive_examples_ = len(positive_pairs)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("Le modèle NCF doit être entraîné avec fit()")

    def recommend_ids(self, user: str, top_n: int = 10) -> list[str]:
        return self.recommend(user, top_n=top_n)[self.artist_col].tolist()

    def recommend(self, user: str, top_n: int = 10) -> pd.DataFrame:
        self._check_fitted()
        if user not in self.user_to_id_:
            raise KeyError(f"Utilisateur inconnu : {user}")
        candidates = sorted(self.catalog_ - self.known_artists_[user])
        if not candidates:
            return pd.DataFrame(columns=["rank", self.artist_col, self.display_col, "score"])

        user_id = self.user_to_id_[user]
        candidate_ids = [self.artist_to_id_[artist] for artist in candidates]
        user_tensor = torch.full(
            (len(candidate_ids),), user_id, dtype=torch.long, device=self.device
        )
        artist_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=self.device)
        self.model_.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model_(user_tensor, artist_tensor))
        count = min(top_n, len(candidates))
        scores, indices = torch.topk(probabilities, k=count)
        selected = [candidates[index] for index in indices.cpu().tolist()]
        frame = pd.DataFrame(
            {
                "rank": range(1, count + 1),
                self.artist_col: selected,
                self.display_col: [self.artist_names_.get(a, a) for a in selected],
                "score": scores.cpu().tolist(),
            }
        )
        return frame


@dataclass
class NCFExperiment:
    recommender: NCFArtistRecommender
    dataset_version: str
    best_config: str
    tuning_results: pd.DataFrame
    validation_metrics: dict[str, float | int]
    test_metrics: dict[str, float | int]
    split_sizes: dict[str, int]


def select_and_fit_best_ncf(
    listens: pd.DataFrame,
    configs: Iterable[dict] = DEFAULT_NCF_CONFIGS,
    *,
    epochs: int = 5,
    top_n: int = 10,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    device: str | torch.device = "auto",
    verbose: bool = False,
) -> NCFExperiment:
    """Sélectionne une configuration NCF puis entraîne le réseau final."""
    dataset_version = get_dataset_version(listens)
    train_df, validation_df, test_df = chronological_split(
        listens,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    configs = [dict(config) for config in configs]
    if not configs:
        raise ValueError("Au moins une configuration NCF est nécessaire")

    results: list[dict] = []
    config_by_name: dict[str, dict] = {}
    for index, config in enumerate(configs, start=1):
        name = str(config.pop("name", f"config_{index}"))
        config_by_name[name] = dict(config)
        if verbose:
            print(f"NCF {index}/{len(configs)} — {name}", flush=True)
        recommender = NCFArtistRecommender(
            **config,
            epochs=epochs,
            random_state=random_state,
            device=device,
            verbose=verbose,
        ).fit(train_df)
        metrics = evaluate_recommender(recommender, validation_df, top_n=top_n)
        results.append(
            {
                "config": name,
                **config,
                "final_loss": recommender.training_losses_[-1],
                **metrics,
            }
        )
        if verbose:
            print(
                f"  Recall@{top_n}={metrics[f'recall@{top_n}']:.4f} | "
                f"NDCG@{top_n}={metrics[f'ndcg@{top_n}']:.4f}",
                flush=True,
            )

    results_df = pd.DataFrame(results)
    ndcg_column = f"ndcg@{top_n}"
    recall_column = f"recall@{top_n}"
    ranked = results_df.sort_values(
        [ndcg_column, recall_column, "config"],
        ascending=[False, False, True],
        na_position="last",
        kind="stable",
    )
    if ranked[ndcg_column].isna().all():
        raise ValueError("Aucun utilisateur évaluable pour sélectionner NCF")
    best_row = ranked.iloc[0]
    best_name = str(best_row["config"])

    final_train = pd.concat([train_df, validation_df], ignore_index=True)
    final_recommender = NCFArtistRecommender(
        **config_by_name[best_name],
        epochs=epochs,
        random_state=random_state,
        device=device,
        verbose=verbose,
    ).fit(final_train)
    test_metrics = evaluate_recommender(final_recommender, test_df, top_n=top_n)
    validation_metrics = {
        recall_column: float(best_row[recall_column]),
        ndcg_column: float(best_row[ndcg_column]),
        "n_evaluable_users": int(best_row["n_evaluable_users"]),
    }
    if verbose:
        print(f"Meilleure configuration : {best_name}", flush=True)

    return NCFExperiment(
        recommender=final_recommender,
        dataset_version=dataset_version,
        best_config=best_name,
        tuning_results=results_df,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        split_sizes={
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df),
        },
    )
