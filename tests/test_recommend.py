import unittest

import pandas as pd

from src.recommend import (
    KMeansArtistRecommender,
    chronological_split,
    select_and_fit_best_kmeans,
)


def synthetic_listens() -> pd.DataFrame:
    rows = []
    known_patterns = [
        ("a", "b", "c"),
        ("a", "c", "b"),
        ("b", "c", "a"),
        ("x", "y", "z"),
        ("x", "z", "y"),
        ("y", "z", "x"),
    ]
    for user_index, (first, second, discovery) in enumerate(known_patterns):
        user = f"user-{user_index}"
        artists = [first, second, first, second, first, discovery, second, discovery]
        for timestamp, artist in enumerate(artists, start=1):
            rows.append(
                {
                    "user": user,
                    "artist_uid": f"mbid:{artist}",
                    "artist": artist.upper(),
                    "artist_credit_weight": 1.0,
                    "listened_at": timestamp,
                }
            )
    return pd.DataFrame(rows)


class KMeansRecommendationTests(unittest.TestCase):
    def test_split_keeps_a_collaboration_timestamp_together(self):
        listens = synthetic_listens()
        duplicate_credit = listens.iloc[[0]].copy()
        duplicate_credit["artist_uid"] = "mbid:guest"
        listens = pd.concat([listens, duplicate_credit], ignore_index=True)

        train, validation, test = chronological_split(
            listens,
            train_ratio=0.6,
            validation_ratio=0.2,
        )

        locations = sum(
            int(((part["user"] == "user-0") & (part["listened_at"] == 1)).any())
            for part in (train, validation, test)
        )
        self.assertEqual(locations, 1)
        self.assertEqual(
            len(train[(train["user"] == "user-0") & (train["listened_at"] == 1)]),
            2,
        )

    def test_recommendations_exclude_known_artists(self):
        listens = synthetic_listens()
        recommender = KMeansArtistRecommender(n_clusters=2).fit(listens)

        recommendations = recommender.recommend_ids("user-0", top_n=3)

        self.assertTrue(
            set(recommendations).isdisjoint(recommender.known_artists_["user-0"])
        )

    def test_selects_k_and_fits_the_final_model(self):
        experiment = select_and_fit_best_kmeans(
            synthetic_listens(),
            k_values=[2, 3],
            top_n=2,
            train_ratio=0.6,
            validation_ratio=0.2,
        )

        self.assertIn(experiment.best_k, {2, 3})
        self.assertEqual(experiment.dataset_version, "unversioned")
        self.assertEqual(set(experiment.tuning_results["k"]), {2, 3})
        self.assertTrue(hasattr(experiment.recommender, "model_"))
        self.assertGreater(
            experiment.validation_metrics["n_evaluable_users"], 0
        )


if __name__ == "__main__":
    unittest.main()
