"""Pipeline réutilisable de recommandation d'artistes avec XGBoost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier

from .recommend import (
    DEFAULT_ARTIST_COLUMN,
    DEFAULT_DISPLAY_COLUMN,
    chronological_split,
    evaluate_recommender,
    get_dataset_version,
)


FEATURE_COLUMNS = [
    "user_activity",
    "artist_popularity",
    "user_diversity",
    "neighbor_affinity_k5",
    "neighbor_affinity_k20",
]

DEFAULT_XGBOOST_CONFIGS = (
    {"name": "balanced", "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
    {"name": "shallow", "n_estimators": 150, "max_depth": 2, "learning_rate": 0.05},
    {"name": "regularized", "n_estimators": 200, "max_depth": 3, "learning_rate": 0.03},
)


def _history_future_split(
    listens: pd.DataFrame,
    history_ratio: float,
    *,
    user_col: str,
    time_col: str = "listened_at",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crée les exemples supervisés sans séparer un même timestamp."""
    if not 0 < history_ratio < 1:
        raise ValueError("history_ratio doit être compris entre 0 et 1")

    history_parts: list[pd.DataFrame] = []
    future_parts: list[pd.DataFrame] = []
    ordered = listens.sort_values([user_col, time_col], kind="stable")
    for _, group in ordered.groupby(user_col, sort=False):
        timestamps = group[time_col].drop_duplicates().tolist()
        if len(timestamps) < 2:
            continue
        split_index = min(max(1, int(len(timestamps) * history_ratio)), len(timestamps) - 1)
        history_times = set(timestamps[:split_index])
        history_parts.append(group[group[time_col].isin(history_times)])
        future_parts.append(group[~group[time_col].isin(history_times)])

    if not history_parts:
        raise ValueError("Pas assez d'historique pour créer les exemples XGBoost")
    return (
        pd.concat(history_parts, ignore_index=True),
        pd.concat(future_parts, ignore_index=True),
    )


class XGBoostArtistRecommender:
    """Classe les artistes candidats avec popularité et affinité de voisinage."""

    def __init__(
        self,
        *,
        model_params: dict | None = None,
        history_ratio: float = 0.6,
        negative_ratio: int = 3,
        random_state: int = 42,
        user_col: str = "user",
        artist_col: str = DEFAULT_ARTIST_COLUMN,
        display_col: str = DEFAULT_DISPLAY_COLUMN,
    ) -> None:
        if negative_ratio < 1:
            raise ValueError("negative_ratio doit être positif")
        self.model_params = dict(model_params or {})
        self.history_ratio = history_ratio
        self.negative_ratio = negative_ratio
        self.random_state = random_state
        self.user_col = user_col
        self.artist_col = artist_col
        self.display_col = display_col

    def _set_context(self, listens: pd.DataFrame) -> None:
        self.context_listens_ = listens.copy()
        self.known_artists_ = (
            listens.groupby(self.user_col)[self.artist_col].apply(set).to_dict()
        )
        self.catalog_ = set(listens[self.artist_col].dropna().unique())
        self.user_activity_ = (
            listens.groupby(self.user_col)["recording_msid"].nunique()
            if "recording_msid" in listens.columns
            else listens.groupby(self.user_col).size()
        )
        self.user_diversity_ = listens.groupby(self.user_col)[self.artist_col].nunique()
        self.artist_popularity_ = listens.groupby(self.artist_col)[self.user_col].nunique()
        self.user_artist_matrix_ = (
            pd.crosstab(listens[self.user_col], listens[self.artist_col]) > 0
        ).astype(float)
        similarity = cosine_similarity(self.user_artist_matrix_)
        self.user_similarity_ = pd.DataFrame(
            similarity,
            index=self.user_artist_matrix_.index,
            columns=self.user_artist_matrix_.index,
        )
        self.global_ranking_ = self.artist_popularity_.sort_values(
            ascending=False, kind="stable"
        ).index.tolist()
        if self.display_col in listens.columns:
            self.artist_names_ = (
                listens[[self.artist_col, self.display_col]]
                .drop_duplicates(self.artist_col)
                .set_index(self.artist_col)[self.display_col]
                .to_dict()
            )
        else:
            self.artist_names_ = {}
        self._affinity_cache: dict[tuple[str, int], pd.Series] = {}

    def _neighbor_affinity(self, user: str, k: int) -> pd.Series:
        cache_key = (user, k)
        if cache_key in self._affinity_cache:
            return self._affinity_cache[cache_key]

        similarities = self.user_similarity_.loc[user].drop(user)
        neighbors = similarities[similarities > 0].nlargest(k)
        if neighbors.empty:
            affinity = pd.Series(0.0, index=self.user_artist_matrix_.columns)
        else:
            interactions = self.user_artist_matrix_.loc[neighbors.index]
            affinity = (neighbors @ interactions) / neighbors.sum()
        self._affinity_cache[cache_key] = affinity
        return affinity

    def _candidate_features(self, user: str, artists: Iterable[str]) -> pd.DataFrame:
        artists = list(artists)
        frame = pd.DataFrame({self.artist_col: artists})
        frame["user_activity"] = float(self.user_activity_.get(user, 0))
        frame["artist_popularity"] = (
            frame[self.artist_col].map(self.artist_popularity_).fillna(0).astype(float)
        )
        frame["user_diversity"] = float(self.user_diversity_.get(user, 0))
        for k in (5, 20):
            affinity = self._neighbor_affinity(user, k)
            frame[f"neighbor_affinity_k{k}"] = (
                frame[self.artist_col].map(affinity).fillna(0.0).astype(float)
            )
        return frame

    def _supervised_examples(
        self,
        history: pd.DataFrame,
        future: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(self.random_state)
        history_artists = history.groupby(self.user_col)[self.artist_col].apply(set)
        future_artists = future.groupby(self.user_col)[self.artist_col].apply(set)
        rows: list[dict] = []

        for user, known in history_artists.items():
            future_for_user = future_artists.get(user, set())
            positives = sorted((future_for_user - known) & self.catalog_)
            for artist in positives:
                rows.append({self.user_col: user, self.artist_col: artist, "target": 1})

            negative_candidates = sorted(self.catalog_ - known - future_for_user)
            negative_count = min(len(negative_candidates), len(positives) * self.negative_ratio)
            if negative_count:
                sampled = rng.choice(negative_candidates, size=negative_count, replace=False)
                for artist in sampled.tolist():
                    rows.append({self.user_col: user, self.artist_col: artist, "target": 0})

        supervised = pd.DataFrame(rows)
        if supervised.empty or supervised["target"].nunique() < 2:
            raise ValueError(
                "Impossible de créer des exemples positifs et négatifs pour XGBoost"
            )

        feature_parts: list[pd.DataFrame] = []
        target_parts: list[pd.Series] = []
        for user, group in supervised.groupby(self.user_col, sort=False):
            feature_parts.append(
                self._candidate_features(user, group[self.artist_col])[FEATURE_COLUMNS]
            )
            target_parts.append(group["target"].reset_index(drop=True))
        return (
            pd.concat(feature_parts, ignore_index=True),
            pd.concat(target_parts, ignore_index=True),
        )

    def fit(self, listens: pd.DataFrame) -> "XGBoostArtistRecommender":
        required = {self.user_col, self.artist_col, "listened_at"}
        missing = required.difference(listens.columns)
        if missing:
            raise ValueError(f"Colonnes absentes : {', '.join(sorted(missing))}")

        history, future = _history_future_split(
            listens,
            self.history_ratio,
            user_col=self.user_col,
        )
        self._set_context(history)
        features, targets = self._supervised_examples(history, future)

        params = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": self.random_state,
            "n_jobs": 1,
            "tree_method": "hist",
        }
        params.update(self.model_params)
        self.model_ = XGBClassifier(**params)
        self.model_.fit(features, targets)
        self.training_examples_ = len(features)
        self.positive_examples_ = int(targets.sum())
        self._set_context(listens)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("Le modèle XGBoost doit être entraîné avec fit()")

    def recommend_ids(self, user: str, top_n: int = 10) -> list[str]:
        return self.recommend(user, top_n=top_n)[self.artist_col].tolist()

    def recommend(self, user: str, top_n: int = 10) -> pd.DataFrame:
        self._check_fitted()
        if user not in self.known_artists_:
            raise KeyError(f"Utilisateur inconnu : {user}")
        candidates = sorted(self.catalog_ - self.known_artists_[user])
        if not candidates:
            return pd.DataFrame(columns=["rank", self.artist_col, self.display_col, "score"])
        candidate_frame = self._candidate_features(user, candidates)
        candidate_frame["score"] = self.model_.predict_proba(
            candidate_frame[FEATURE_COLUMNS]
        )[:, 1]
        top = candidate_frame.nlargest(min(top_n, len(candidate_frame)), "score").copy()
        top.insert(0, "rank", range(1, len(top) + 1))
        top[self.display_col] = top[self.artist_col].map(self.artist_names_).fillna(
            top[self.artist_col]
        )
        return top[["rank", self.artist_col, self.display_col, "score"]].reset_index(drop=True)

    def feature_importance(self) -> pd.Series:
        self._check_fitted()
        return pd.Series(
            self.model_.feature_importances_, index=FEATURE_COLUMNS, name="importance"
        ).sort_values(ascending=False)


@dataclass
class XGBoostExperiment:
    recommender: XGBoostArtistRecommender
    dataset_version: str
    best_config: str
    tuning_results: pd.DataFrame
    validation_metrics: dict[str, float | int]
    test_metrics: dict[str, float | int]
    split_sizes: dict[str, int]


def select_and_fit_best_xgboost(
    listens: pd.DataFrame,
    configs: Iterable[dict] = DEFAULT_XGBOOST_CONFIGS,
    *,
    top_n: int = 10,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    verbose: bool = False,
) -> XGBoostExperiment:
    """Sélectionne une configuration sur validation puis entraîne le modèle final."""
    dataset_version = get_dataset_version(listens)
    train_df, validation_df, test_df = chronological_split(
        listens,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    configs = [dict(config) for config in configs]
    if not configs:
        raise ValueError("Au moins une configuration XGBoost est nécessaire")

    results: list[dict] = []
    for index, config in enumerate(configs, start=1):
        name = str(config.pop("name", f"config_{index}"))
        if verbose:
            print(f"XGBoost {index}/{len(configs)} — {name}", flush=True)
        recommender = XGBoostArtistRecommender(
            model_params=config,
            random_state=random_state,
        ).fit(train_df)
        metrics = evaluate_recommender(recommender, validation_df, top_n=top_n)
        results.append({"config": name, **config, **metrics})
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
        raise ValueError("Aucun utilisateur évaluable pour sélectionner XGBoost")
    best_row = ranked.iloc[0]
    best_name = str(best_row["config"])

    # Retrouve les paramètres depuis les résultats, sans les colonnes de métriques.
    metric_columns = {"config", recall_column, ndcg_column, "n_evaluable_users"}
    best_params = {
        key: best_row[key]
        for key in results_df.columns
        if key not in metric_columns and pd.notna(best_row[key])
    }
    for key in ("n_estimators", "max_depth"):
        if key in best_params:
            best_params[key] = int(best_params[key])

    final_train = pd.concat([train_df, validation_df], ignore_index=True)
    final_recommender = XGBoostArtistRecommender(
        model_params=best_params,
        random_state=random_state,
    ).fit(final_train)
    test_metrics = evaluate_recommender(final_recommender, test_df, top_n=top_n)
    validation_metrics = {
        recall_column: float(best_row[recall_column]),
        ndcg_column: float(best_row[ndcg_column]),
        "n_evaluable_users": int(best_row["n_evaluable_users"]),
    }
    if verbose:
        print(f"Meilleure configuration : {best_name}", flush=True)

    return XGBoostExperiment(
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
