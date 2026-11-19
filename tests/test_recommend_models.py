import unittest

import pandas as pd

from src.recommend_deep_learning import select_and_fit_best_ncf
from src.recommend_xgboost import select_and_fit_best_xgboost


def synthetic_model_listens() -> pd.DataFrame:
    patterns = [
        ("a", "b", "c", "d"),
        ("c", "d", "e", "f"),
        ("e", "f", "a", "b"),
        ("b", "c", "d", "e"),
    ] * 2
    rows = []
    for user_index, (first, second, discovery, validation_artist) in enumerate(
        patterns
    ):
        sequence = [
            first,
            second,
            first,
            second,
            first,
            discovery,
            first,
            second,
            discovery,
            validation_artist,
            validation_artist,
            second,
        ]
        for hour, artist in enumerate(sequence, start=1):
            rows.append(
                {
                    "user": f"user-{user_index}",
                    "artist_uid": f"mbid:{artist}",
                    "artist": artist.upper(),
                    "listened_at": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(hours=hour),
                    "recording_msid": f"recording-{user_index}-{hour}",
                    "dataset_version": "test-v1",
                }
            )
    return pd.DataFrame(rows)


class AdvancedRecommendationTests(unittest.TestCase):
    def test_xgboost_pipeline_returns_ranked_unknown_artists(self):
        experiment = select_and_fit_best_xgboost(
            synthetic_model_listens(),
            configs=[
                {
                    "name": "tiny",
                    "n_estimators": 10,
                    "max_depth": 2,
                    "learning_rate": 0.1,
                }
            ],
            top_n=2,
            train_ratio=0.75,
            validation_ratio=0.125,
        )

        recommendations = experiment.recommender.recommend("user-0", top_n=2)

        self.assertEqual(experiment.dataset_version, "test-v1")
        self.assertEqual(experiment.best_config, "tiny")
        self.assertEqual(len(recommendations), 2)
        self.assertTrue(
            set(recommendations["artist_uid"]).isdisjoint(
                experiment.recommender.known_artists_["user-0"]
            )
        )

    def test_ncf_pipeline_returns_ranked_unknown_artists(self):
        experiment = select_and_fit_best_ncf(
            synthetic_model_listens(),
            configs=[
                {
                    "name": "tiny",
                    "embedding_dim": 4,
                    "weight_decay": 0.0,
                }
            ],
            epochs=1,
            top_n=2,
            train_ratio=0.75,
            validation_ratio=0.125,
            device="cpu",
        )

        recommendations = experiment.recommender.recommend("user-0", top_n=2)

        self.assertEqual(experiment.dataset_version, "test-v1")
        self.assertEqual(experiment.best_config, "tiny")
        self.assertEqual(len(recommendations), 2)
        self.assertTrue(
            set(recommendations["artist_uid"]).isdisjoint(
                experiment.recommender.known_artists_["user-0"]
            )
        )


if __name__ == "__main__":
    unittest.main()
