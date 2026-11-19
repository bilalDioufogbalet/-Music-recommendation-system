"""Recommandation d'artistes par clustering K-Means.

Toute la logique réutilisable vit ici afin que le notebook K-Means reste une
courte démonstration : découpage temporel, sélection de ``k``, entraînement,
recommandation et évaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


DEFAULT_ARTIST_COLUMN = "artist_uid"
DEFAULT_DISPLAY_COLUMN = "artist"


def _validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns).difference(df.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Colonnes indispensables absentes : {names}")


def get_dataset_version(listens: pd.DataFrame) -> str:
    """Retourne l'unique version du dataset ou refuse un mélange de versions."""
    if "dataset_version" not in listens.columns:
        return "unversioned"
    dataset_versions = listens["dataset_version"].dropna().unique().tolist()
    if len(dataset_versions) != 1:
        raise ValueError(
            "Une expérience ne peut pas mélanger plusieurs versions de données"
        )
    return str(dataset_versions[0])


def chronological_split(
    listens: pd.DataFrame,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    user_col: str = "user",
    time_col: str = "listened_at",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sépare chronologiquement chaque utilisateur en train/validation/test.

    Toutes les lignes partageant le même timestamp restent dans la même partie.
    C'est important depuis que les collaborations produisent une ligne par
    artiste crédité.
    """
    _validate_columns(listens, [user_col, time_col])
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("Les ratios train et validation doivent être positifs")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Il faut conserver une proportion positive pour le test")

    ordered = listens.sort_values([user_col, time_col], kind="stable").copy()
    parts: dict[str, list[pd.DataFrame]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for _, user_listens in ordered.groupby(user_col, sort=False):
        timestamps = user_listens[time_col].drop_duplicates().tolist()
        event_count = len(timestamps)
        if event_count < 3:
            raise ValueError(
                "Chaque utilisateur doit avoir au moins trois timestamps distincts"
            )

        train_end = max(1, int(event_count * train_ratio))
        validation_end = max(
            train_end + 1,
            int(event_count * (train_ratio + validation_ratio)),
        )
        validation_end = min(validation_end, event_count - 1)
        train_end = min(train_end, validation_end - 1)

        train_times = set(timestamps[:train_end])
        validation_times = set(timestamps[train_end:validation_end])

        parts["train"].append(
            user_listens[user_listens[time_col].isin(train_times)]
        )
        parts["validation"].append(
            user_listens[user_listens[time_col].isin(validation_times)]
        )
        parts["test"].append(
            user_listens[
                ~user_listens[time_col].isin(train_times | validation_times)
            ]
        )

    return tuple(
        pd.concat(parts[name], ignore_index=True)
        for name in ("train", "validation", "test")
    )


def recall_at_k(recommendations: Iterable[str], relevant: set[str]) -> float:
    """Calcule le Recall@k pour une liste déjà limitée à k résultats."""
    if not relevant:
        raise ValueError("Le Recall ne peut pas être calculé sans élément pertinent")
    return len(set(recommendations) & relevant) / len(relevant)


def ndcg_at_k(recommendations: Iterable[str], relevant: set[str]) -> float:
    """Calcule le NDCG@k avec une pertinence binaire."""
    recommendations = list(recommendations)
    if not relevant:
        raise ValueError("Le NDCG ne peut pas être calculé sans élément pertinent")

    dcg = sum(
        1 / np.log2(rank + 2)
        for rank, artist in enumerate(recommendations)
        if artist in relevant
    )
    ideal_hits = min(len(relevant), len(recommendations))
    idcg = sum(1 / np.log2(rank + 2) for rank in range(ideal_hits))
    return float(dcg / idcg) if idcg else 0.0


class KMeansArtistRecommender:
    """Recommande les artistes populaires du cluster d'un utilisateur."""

    def __init__(
        self,
        n_clusters: int,
        *,
        random_state: int = 42,
        n_init: int = 10,
        user_col: str = "user",
        artist_col: str = DEFAULT_ARTIST_COLUMN,
        display_col: str = DEFAULT_DISPLAY_COLUMN,
        interaction_weight_col: str = "artist_credit_weight",
        fallback_to_global: bool = True,
    ) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters doit être supérieur ou égal à 2")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.user_col = user_col
        self.artist_col = artist_col
        self.display_col = display_col
        self.interaction_weight_col = interaction_weight_col
        self.fallback_to_global = fallback_to_global

    def fit(self, listens: pd.DataFrame) -> "KMeansArtistRecommender":
        """Entraîne K-Means et construit les classements d'artistes."""
        _validate_columns(listens, [self.user_col, self.artist_col])
        if listens.empty:
            raise ValueError("Impossible d'entraîner le modèle sur un DataFrame vide")

        user_count = listens[self.user_col].nunique()
        if self.n_clusters > user_count:
            raise ValueError(
                f"n_clusters={self.n_clusters} dépasse le nombre d'utilisateurs "
                f"({user_count})"
            )

        group_columns = [self.user_col, self.artist_col]
        if self.interaction_weight_col in listens.columns:
            interactions = (
                listens.groupby(group_columns, as_index=False)[
                    self.interaction_weight_col
                ]
                .sum()
                .rename(columns={self.interaction_weight_col: "interaction_count"})
            )
        else:
            interactions = (
                listens.groupby(group_columns)
                .size()
                .rename("interaction_count")
                .reset_index()
            )

        user_artist_matrix = interactions.pivot(
            index=self.user_col,
            columns=self.artist_col,
            values="interaction_count",
        ).fillna(0.0)
        normalized_matrix = normalize(user_artist_matrix, norm="l2")

        model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        labels = model.fit_predict(normalized_matrix)
        user_clusters = pd.Series(
            labels,
            index=user_artist_matrix.index,
            name="cluster",
        )

        cluster_frame = user_clusters.rename_axis(self.user_col).reset_index()
        cluster_interactions = interactions.merge(
            cluster_frame,
            on=self.user_col,
            how="left",
            validate="many_to_one",
        )
        cluster_popularity = (
            cluster_interactions.groupby(["cluster", self.artist_col])[
                self.user_col
            ]
            .nunique()
            .rename("n_users")
            .reset_index()
            .sort_values(
                ["cluster", "n_users", self.artist_col],
                ascending=[True, False, True],
                kind="stable",
            )
        )

        self.model_ = model
        self.interactions_ = interactions
        self.user_artist_matrix_ = user_artist_matrix
        self.user_clusters_ = user_clusters
        self.cluster_popularity_ = cluster_popularity
        self.cluster_rankings_ = (
            cluster_popularity.groupby("cluster")[self.artist_col]
            .apply(list)
            .to_dict()
        )
        self.known_artists_ = (
            listens.groupby(self.user_col)[self.artist_col].apply(set).to_dict()
        )
        self.catalog_ = set(user_artist_matrix.columns)
        self.global_ranking_ = (
            listens.groupby(self.artist_col)[self.user_col]
            .nunique()
            .sort_values(ascending=False, kind="stable")
            .index.tolist()
        )

        if self.display_col in listens.columns:
            self.artist_names_ = (
                listens[[self.artist_col, self.display_col]]
                .drop_duplicates(self.artist_col)
                .set_index(self.artist_col)[self.display_col]
                .to_dict()
            )
        else:
            self.artist_names_ = {}

        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("Le modèle doit être entraîné avec fit()")

    def recommend_ids(self, user: str, top_n: int = 10) -> list[str]:
        """Retourne les identités d'artistes recommandées, sans artistes connus."""
        self._check_is_fitted()
        if top_n <= 0:
            raise ValueError("top_n doit être positif")
        if user not in self.known_artists_:
            raise KeyError(f"Utilisateur inconnu : {user}")

        cluster = int(self.user_clusters_.loc[user])
        known = self.known_artists_[user]
        candidates = list(self.cluster_rankings_.get(cluster, []))
        if self.fallback_to_global:
            candidates.extend(self.global_ranking_)

        recommendations: list[str] = []
        seen: set[str] = set()
        for artist in candidates:
            if artist in known or artist in seen:
                continue
            recommendations.append(artist)
            seen.add(artist)
            if len(recommendations) == top_n:
                break
        return recommendations

    def recommend(self, user: str, top_n: int = 10) -> pd.DataFrame:
        """Retourne une table lisible de recommandations pour un utilisateur."""
        artist_ids = self.recommend_ids(user, top_n=top_n)
        return pd.DataFrame(
            {
                "rank": range(1, len(artist_ids) + 1),
                self.artist_col: artist_ids,
                self.display_col: [
                    self.artist_names_.get(artist, artist) for artist in artist_ids
                ],
            }
        )

    def cluster_sizes(self) -> pd.Series:
        """Retourne le nombre d'utilisateurs dans chaque cluster."""
        self._check_is_fitted()
        return self.user_clusters_.value_counts().sort_index()


def relevant_artists_by_user(
    recommender: KMeansArtistRecommender,
    future_listens: pd.DataFrame,
) -> dict[str, set[str]]:
    """Construit les découvertes futures évaluables pour chaque utilisateur."""
    _validate_columns(
        future_listens,
        [recommender.user_col, recommender.artist_col],
    )
    future_artists = (
        future_listens.groupby(recommender.user_col)[recommender.artist_col]
        .apply(set)
        .to_dict()
    )

    relevant: dict[str, set[str]] = {}
    for user, artists in future_artists.items():
        if user not in recommender.known_artists_:
            continue
        discoveries = (
            artists
            .difference(recommender.known_artists_[user])
            .intersection(recommender.catalog_)
        )
        if discoveries:
            relevant[user] = discoveries
    return relevant


def evaluate_recommender(
    recommender: KMeansArtistRecommender,
    future_listens: pd.DataFrame,
    *,
    top_n: int = 10,
) -> dict[str, float | int]:
    """Évalue un modèle sur les nouvelles écoutes d'une période future."""
    relevant_by_user = relevant_artists_by_user(recommender, future_listens)
    if not relevant_by_user:
        return {
            f"recall@{top_n}": float("nan"),
            f"ndcg@{top_n}": float("nan"),
            "n_evaluable_users": 0,
        }

    recalls: list[float] = []
    ndcgs: list[float] = []
    for user, relevant in relevant_by_user.items():
        recommendations = recommender.recommend_ids(user, top_n=top_n)
        recalls.append(recall_at_k(recommendations, relevant))
        ndcgs.append(ndcg_at_k(recommendations, relevant))

    return {
        f"recall@{top_n}": float(np.mean(recalls)),
        f"ndcg@{top_n}": float(np.mean(ndcgs)),
        "n_evaluable_users": len(relevant_by_user),
    }


def tune_kmeans(
    train_listens: pd.DataFrame,
    validation_listens: pd.DataFrame,
    k_values: Iterable[int],
    *,
    top_n: int = 10,
    random_state: int = 42,
    artist_col: str = DEFAULT_ARTIST_COLUMN,
    display_col: str = DEFAULT_DISPLAY_COLUMN,
    verbose: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Teste plusieurs valeurs de k et choisit celle au meilleur NDCG."""
    user_count = train_listens["user"].nunique()
    valid_k_values = sorted(
        {int(k) for k in k_values if 2 <= int(k) <= user_count}
    )
    if not valid_k_values:
        raise ValueError("Aucune valeur de k n'est compatible avec les données")

    results: list[dict[str, float | int]] = []
    for position, k in enumerate(valid_k_values, start=1):
        if verbose:
            print(
                f"K-Means {position}/{len(valid_k_values)} — entraînement avec k={k}",
                flush=True,
            )
        recommender = KMeansArtistRecommender(
            n_clusters=k,
            random_state=random_state,
            artist_col=artist_col,
            display_col=display_col,
        ).fit(train_listens)
        metrics = evaluate_recommender(
            recommender,
            validation_listens,
            top_n=top_n,
        )
        results.append({"k": k, **metrics})
        if verbose:
            print(
                f"  Recall@{top_n}={metrics[f'recall@{top_n}']:.4f} | "
                f"NDCG@{top_n}={metrics[f'ndcg@{top_n}']:.4f}",
                flush=True,
            )

    results_df = pd.DataFrame(results)
    ndcg_column = f"ndcg@{top_n}"
    recall_column = f"recall@{top_n}"
    if results_df[ndcg_column].isna().all():
        raise ValueError(
            "Aucun utilisateur évaluable dans la validation : impossible de choisir k"
        )

    ranked = results_df.sort_values(
        [ndcg_column, recall_column, "k"],
        ascending=[False, False, True],
        na_position="last",
        kind="stable",
    )
    best_k = int(ranked.iloc[0]["k"])
    return results_df, best_k


@dataclass
class KMeansExperiment:
    """Résultat complet d'une sélection de k et d'un entraînement final."""

    recommender: KMeansArtistRecommender
    dataset_version: str
    best_k: int
    tuning_results: pd.DataFrame
    validation_metrics: dict[str, float | int]
    test_metrics: dict[str, float | int]
    split_sizes: dict[str, int]


def select_and_fit_best_kmeans(
    listens: pd.DataFrame,
    k_values: Iterable[int] = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20),
    *,
    top_n: int = 10,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    artist_col: str = DEFAULT_ARTIST_COLUMN,
    display_col: str = DEFAULT_DISPLAY_COLUMN,
    verbose: bool = False,
) -> KMeansExperiment:
    """Sélectionne le meilleur k puis entraîne directement le modèle final.

    Le meilleur ``k`` est choisi sur la validation par NDCG, puis Recall et
    enfin la plus petite valeur de ``k`` en cas d'égalité. Le modèle final est
    réentraîné sur train + validation et évalué une seule fois sur le test.
    """
    dataset_version = get_dataset_version(listens)

    train_df, validation_df, test_df = chronological_split(
        listens,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    if verbose:
        print(
            f"Dataset {dataset_version} — "
            f"train={len(train_df)}, validation={len(validation_df)}, "
            f"test={len(test_df)}",
            flush=True,
        )
    tuning_results, best_k = tune_kmeans(
        train_df,
        validation_df,
        k_values,
        top_n=top_n,
        random_state=random_state,
        artist_col=artist_col,
        display_col=display_col,
        verbose=verbose,
    )

    validation_metrics = (
        tuning_results.loc[tuning_results["k"].eq(best_k)]
        .iloc[0]
        .drop(labels="k")
        .to_dict()
    )
    final_train_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_recommender = KMeansArtistRecommender(
        n_clusters=best_k,
        random_state=random_state,
        artist_col=artist_col,
        display_col=display_col,
    ).fit(final_train_df)
    test_metrics = evaluate_recommender(
        final_recommender,
        test_df,
        top_n=top_n,
    )
    if verbose:
        print(
            f"Meilleur k={best_k} — modèle final entraîné et évalué sur le test",
            flush=True,
        )

    return KMeansExperiment(
        recommender=final_recommender,
        dataset_version=dataset_version,
        best_k=best_k,
        tuning_results=tuning_results,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        split_sizes={
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df),
        },
    )
