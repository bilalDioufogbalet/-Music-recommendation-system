import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.clean import (
    build_processed_dataset,
    build_user_quality_report,
    clean_listens,
    expand_artist_identities,
    filter_usable_users,
)


def raw_listen(
    artist_credit,
    timestamp,
    artist_credits=None,
    user="Alice",
    track="Example track",
):
    return {
        "user": user,
        "artist_credit": artist_credit,
        "artist_credits": artist_credits or [],
        "track": track,
        "release": None,
        "listened_at": timestamp,
        "recording_msid": None,
    }


class CleanListensTests(unittest.TestCase):
    def test_removes_duplicates_but_keeps_real_relistens(self):
        listens = pd.DataFrame(
            [
                raw_listen("Little Simz", 1_700_000_000),
                raw_listen(" LITTLE   SIMZ ", 1_700_000_000),
                raw_listen("Little Simz", 1_700_000_300),
            ]
        )

        cleaned, duplicate_count = clean_listens(listens)

        self.assertEqual(duplicate_count, 1)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned.iloc[0]["artist_credit"], "Little Simz")

    def test_keeps_different_tracks_at_the_same_timestamp(self):
        listens = pd.DataFrame(
            [
                raw_listen("SZA", 1_700_000_000, track="Saturn"),
                raw_listen("SZA", 1_700_000_000, track="Snooze"),
            ]
        )

        cleaned, duplicate_count = clean_listens(listens)

        self.assertEqual(duplicate_count, 0)
        self.assertEqual(len(cleaned), 2)

    def test_expands_a_collaboration_into_unique_mbid_artists(self):
        steve_mbid = "b29020d6-5dbf-4c99-b16f-c009f8a712ff"
        sza_mbid = "272989c8-5535-492d-a25c-9f58803e027f"
        listens = pd.DataFrame(
            [
                raw_listen(
                    "Steve Lacy",
                    1,
                    [{"artist": "Steve Lacy", "artist_mbid": steve_mbid}],
                ),
                raw_listen(
                    "SZA",
                    2,
                    [{"artist": "SZA", "artist_mbid": sza_mbid}],
                ),
                # Aucun mapping sur cette écoute : les deux identités sont
                # néanmoins inférées depuis les crédits fiables ci-dessus.
                raw_listen("Steve Lacy, SZA", 3),
            ]
        )

        expanded = expand_artist_identities(listens)
        collaboration = expanded[expanded["artist_credit"] == "Steve Lacy, SZA"]

        self.assertEqual(len(collaboration), 2)
        self.assertEqual(
            set(collaboration["artist_uid"]),
            {f"mbid:{steve_mbid}", f"mbid:{sza_mbid}"},
        )
        self.assertEqual(
            set(collaboration["artist_identity_source"]), {"inferred_alias"}
        )

    def test_case_variants_share_the_same_identity(self):
        artist_mbid = "d993169f-2033-4810-bae8-564d4aab89cd"
        listens = pd.DataFrame(
            [
                raw_listen(
                    "$NOT",
                    1,
                    [{"artist": "$NOT", "artist_mbid": artist_mbid}],
                ),
                raw_listen("$Not", 2),
            ]
        )

        expanded = expand_artist_identities(listens)

        self.assertEqual(expanded["artist_uid"].nunique(), 1)
        self.assertEqual(set(expanded["artist"]), {"$NOT"})

    def test_removes_duplicates_revealed_after_identity_resolution(self):
        artist_mbid = "b29020d6-5dbf-4c99-b16f-c009f8a712ff"
        listens = pd.DataFrame(
            [
                raw_listen(
                    "Steve Lacy",
                    1,
                    [{"artist": "Steve Lacy", "artist_mbid": artist_mbid}],
                ),
                raw_listen(
                    "Steve Lacy, SZA",
                    1,
                    [{"artist": "Steve Lacy", "artist_mbid": artist_mbid}],
                ),
            ]
        )

        expanded = expand_artist_identities(listens)

        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded.attrs["identity_duplicates_removed"], 1)

    def test_disambiguates_two_artists_with_the_same_display_name(self):
        listens = pd.DataFrame(
            [
                raw_listen(
                    "Zola",
                    1,
                    [{"artist": "Zola", "artist_mbid": "mbid-one"}],
                ),
                raw_listen(
                    "Zola",
                    2,
                    [{"artist": "Zola", "artist_mbid": "mbid-two"}],
                ),
            ]
        )

        expanded = expand_artist_identities(listens)

        self.assertEqual(expanded["artist_uid"].nunique(), 2)
        self.assertEqual(expanded["artist"].nunique(), 2)

    def test_text_fallback_normalizes_unicode_and_typographic_variants(self):
        listens = pd.DataFrame(
            [
                raw_listen("Beyoncé", 1),
                # Même accent, mais stocké sous sa forme Unicode décomposée.
                raw_listen("Beyonce\u0301", 2),
                raw_listen("Jok’Air", 3),
                raw_listen("  jok'air  ", 4),
            ]
        )

        expanded = expand_artist_identities(listens)

        self.assertEqual(expanded.iloc[0]["artist_uid"], "text:beyoncé")
        self.assertEqual(expanded.iloc[0]["artist_uid"], expanded.iloc[1]["artist_uid"])
        self.assertEqual(expanded.iloc[2]["artist_uid"], "text:jok'air")
        self.assertEqual(expanded.iloc[2]["artist_uid"], expanded.iloc[3]["artist_uid"])

    def test_text_fallback_preserves_meaningful_accents(self):
        listens = pd.DataFrame(
            [
                raw_listen("Sơn Tùng M-TP", 1),
                raw_listen("Son Tung M-TP", 2),
            ]
        )

        expanded = expand_artist_identities(listens)

        self.assertEqual(expanded["artist_uid"].nunique(), 2)
        self.assertIn("text:sơn tùng m-tp", set(expanded["artist_uid"]))

    def test_filters_unusable_users_and_reports_each_reason(self):
        rows = []

        def add_event(user, artist_index, hour, event_index):
            rows.append(
                {
                    "user": user,
                    "artist_uid": f"mbid:{user}-{artist_index}",
                    "artist": f"Artist {artist_index}",
                    "track": f"Track {event_index}",
                    "listened_at": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(hours=hour),
                    "recording_msid": f"{user}-{event_index}",
                }
            )

        for index in range(15):
            add_event("valid", index, index * 3, index)
            add_event("short", index, index / 10, index)
        for index in range(14):
            add_event("few", index, index * 3, index)
        for index in range(30):
            artist_index = 0 if index < 16 else index - 15
            add_event("dominant", artist_index, index * 2, index)

        listens = pd.DataFrame(rows)
        report = build_user_quality_report(listens)
        filtered, second_report = filter_usable_users(listens)
        report = report.set_index("user")

        self.assertTrue(report.loc["valid", "is_usable"])
        self.assertEqual(
            report.loc["few", "exclusion_reasons"], "too_few_artists"
        )
        self.assertEqual(
            report.loc["short", "exclusion_reasons"], "history_too_short"
        )
        self.assertEqual(
            report.loc["dominant", "exclusion_reasons"],
            "dominant_artist_over_limit",
        )
        self.assertEqual(set(filtered["user"]), {"valid"})
        self.assertEqual(len(second_report), 4)

    def test_saves_a_versioned_dataset_and_cleaning_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "data" / "raw"
            processed_dir = root / "data" / "processed"
            raw_dir.mkdir(parents=True)
            listens = []
            for index, artist in enumerate(("Alpha", "Beta", "Gamma")):
                listens.append(
                    {
                        "user_name": "listener",
                        "listened_at": 1_700_000_000 + index * 3_600,
                        "recording_msid": f"recording-{index}",
                        "track_metadata": {
                            "artist_name": artist,
                            "track_name": f"Track {index}",
                            "mbid_mapping": {
                                "artist_mbids": [f"mbid-{index}"],
                                "artists": [
                                    {
                                        "artist_credit_name": artist,
                                        "artist_mbid": f"mbid-{index}",
                                        "join_phrase": "",
                                    }
                                ],
                            },
                        },
                    }
                )
            payload = {"payload": {"listens": listens}}
            (raw_dir / "listener_listens.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            result = build_processed_dataset(
                raw_dir,
                processed_dir,
                min_distinct_artists=2,
                max_dominant_artist_share=1.0,
                min_history_span="1h",
            )

            report = json.loads(result.cleaning_report_path.read_text())
            dataset = pd.read_parquet(result.output_path)
            self.assertTrue(result.versioned_output_path.exists())
            self.assertEqual(report["dataset_version"], result.dataset_version)
            self.assertEqual(report["rows"]["raw_loaded"], 3)
            self.assertEqual(report["rows"]["final"], 3)
            self.assertEqual(set(dataset["dataset_version"]), {result.dataset_version})


if __name__ == "__main__":
    unittest.main()
