"""Nettoyage et identification des artistes des écoutes ListenBrainz.

Le pipeline supprime les soumissions en double, puis transforme chaque crédit
d'artiste en une identité individuelle. Les modèles peuvent utiliser
``artist_uid`` comme clé stable et ``artist`` uniquement comme libellé lisible.
"""

from __future__ import annotations

import json
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

try:
    from .paths import PROCESSED_DIR, RAW_DIR
except ImportError:  # Permet aussi l'exécution avec `python src/clean.py`.
    from paths import PROCESSED_DIR, RAW_DIR


RAW_FILE_PATTERN = "*_listens.json"
OUTPUT_FILENAME = "listens.parquet"
QUALITY_REPORT_FILENAME = "user_quality.parquet"
CLEANING_REPORT_FILENAME = "cleaning_report.json"
VERSIONS_DIRECTORY = "versions"
CLEANING_PIPELINE_VERSION = "1.0.0"
DEFAULT_MIN_DISTINCT_ARTISTS = 15
DEFAULT_MAX_DOMINANT_ARTIST_SHARE = 0.5
DEFAULT_MIN_HISTORY_SPAN = "24h"
RAW_TEXT_DEDUPLICATION_COLUMNS = ("user", "artist_credit", "track")
RAW_DEDUPLICATION_COLUMNS = (*RAW_TEXT_DEDUPLICATION_COLUMNS, "listened_at")

_ARTIST_SEPARATOR = re.compile(
    r"\s*(?:,|&|\+|/|;|\bx\b|\bfeat\.?|\bft\.?|\bfeaturing\b|\bwith\b)\s*",
    flags=re.IGNORECASE,
)

_TYPOGRAPHIC_PUNCTUATION = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "`": "'",
        "‐": "-",
        "‑": "-",
        "–": "-",
        "—": "-",
    }
)


@dataclass(frozen=True)
class CleaningResult:
    """Artefacts et compteurs produits par une exécution du nettoyage."""

    output_path: Path
    versioned_output_path: Path
    quality_report_path: Path
    cleaning_report_path: Path
    dataset_version: str
    row_count: int
    duplicate_count: int
    identity_counts: dict[str, int]
    filter_counts: dict[str, int]


def _normalize_text_value(value: str) -> str:
    """Construit une clé Unicode stable sans effacer les accents significatifs.

    NFKC uniformise notamment les caractères pleine largeur et NFC garantit
    qu'un accent composé ou décomposé produit la même clé. Les accents restent
    présents : ``Sơn`` et ``Son`` ne sont pas fusionnés arbitrairement.
    """
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = normalized.translate(_TYPOGRAPHIC_PUNCTUATION).casefold()
    normalized = unicodedata.normalize("NFC", normalized)
    return " ".join(normalized.split())


def _normalized_text(series: pd.Series) -> pd.Series:
    return series.astype("string").map(
        lambda value: (
            pd.NA if pd.isna(value) else _normalize_text_value(str(value))
        )
    )


def _extract_artist_credits(track_metadata: dict) -> list[dict[str, str]]:
    """Extrait les couples nom/MBID individuels fournis par MusicBrainz."""
    mapping = track_metadata.get("mbid_mapping") or {}
    credits: list[dict[str, str]] = []
    seen_mbids: set[str] = set()

    for artist in mapping.get("artists") or []:
        name = artist.get("artist_credit_name")
        mbid = artist.get("artist_mbid")
        if not name or not mbid or mbid in seen_mbids:
            continue
        credits.append({"artist": name, "artist_mbid": mbid})
        seen_mbids.add(mbid)

    # Certains anciens objets ListenBrainz ont un seul MBID sans le détail
    # `artists`. Dans ce cas seulement, le crédit global peut être associé sans
    # ambiguïté à ce MBID.
    if not credits:
        artist_mbids = mapping.get("artist_mbids") or []
        artist_credit = track_metadata.get("artist_name")
        if len(artist_mbids) == 1 and artist_credit:
            credits.append(
                {"artist": artist_credit, "artist_mbid": artist_mbids[0]}
            )

    return credits


def load_raw_listens(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Charge les JSON bruts, à raison d'une ligne par écoute reçue."""
    files = sorted(raw_dir.glob(RAW_FILE_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier correspondant à {RAW_FILE_PATTERN!r} dans {raw_dir}"
        )

    rows: list[dict] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        for listen in data.get("payload", {}).get("listens", []):
            track_metadata = listen.get("track_metadata") or {}
            rows.append(
                {
                    "user": listen.get("user_name"),
                    "artist_credit": track_metadata.get("artist_name"),
                    "artist_credits": _extract_artist_credits(track_metadata),
                    "track": track_metadata.get("track_name"),
                    "release": track_metadata.get("release_name"),
                    "listened_at": listen.get("listened_at"),
                    "recording_msid": listen.get("recording_msid"),
                }
            )

    return pd.DataFrame(rows)


def clean_listens(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Valide, date et déduplique les écoutes avant l'éclatement des artistes."""
    missing_columns = set(RAW_DEDUPLICATION_COLUMNS).difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colonnes indispensables absentes : {missing}")

    cleaned = df.copy()
    if is_datetime64_any_dtype(cleaned["listened_at"]):
        cleaned["listened_at"] = pd.to_datetime(
            cleaned["listened_at"], utc=True, errors="coerce"
        )
    else:
        cleaned["listened_at"] = pd.to_datetime(
            cleaned["listened_at"], unit="s", utc=True, errors="coerce"
        )

    required_columns = list(RAW_DEDUPLICATION_COLUMNS)
    invalid_rows = cleaned[required_columns].isna().any(axis=1)
    for column in RAW_TEXT_DEDUPLICATION_COLUMNS:
        invalid_rows |= _normalized_text(cleaned[column]).eq("")

    if invalid_rows.any():
        count = int(invalid_rows.sum())
        raise ValueError(f"{count} écoute(s) sans utilisateur, artiste, titre ou date")

    temporary_columns: list[str] = []
    for column in RAW_TEXT_DEDUPLICATION_COLUMNS:
        temporary_column = f"__normalized_{column}"
        cleaned[temporary_column] = _normalized_text(cleaned[column])
        temporary_columns.append(temporary_column)

    duplicate_key = [*temporary_columns, "listened_at"]
    duplicate_rows = cleaned.duplicated(subset=duplicate_key, keep="first")
    duplicate_count = int(duplicate_rows.sum())

    cleaned = cleaned.loc[~duplicate_rows].drop(columns=temporary_columns)
    return cleaned.reset_index(drop=True), duplicate_count


def _select_canonical_name(name_counts: Counter[str]) -> str:
    """Choisit un libellé déterministe parmi les variantes d'un même artiste."""
    return min(
        name_counts,
        key=lambda name: (
            -name_counts[name],
            _normalize_text_value(name),
            name,
        ),
    )


def _build_artist_catalog(
    df: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str]]:
    """Construit alias→MBID et MBID→nom canonique depuis les crédits fiables."""
    mbids_by_alias: dict[str, Counter[str]] = defaultdict(Counter)
    names_by_mbid: dict[str, Counter[str]] = defaultdict(Counter)

    for credits in df["artist_credits"]:
        for credit in credits or []:
            name = credit["artist"]
            mbid = credit["artist_mbid"]
            mbids_by_alias[_normalize_text_value(name)][mbid] += 1
            names_by_mbid[mbid][name] += 1

    alias_to_mbid: dict[str, str] = {}
    for alias, mbid_counts in mbids_by_alias.items():
        ranked_mbids = mbid_counts.most_common()
        if len(ranked_mbids) == 1:
            alias_to_mbid[alias] = ranked_mbids[0][0]
            continue

        # Une association très largement majoritaire permet de corriger un
        # mapping isolé manifestement aberrant sans fusionner deux homonymes.
        (dominant_mbid, dominant_count), (_, second_count) = ranked_mbids[:2]
        if dominant_count >= 5 and dominant_count >= second_count * 10:
            alias_to_mbid[alias] = dominant_mbid

    canonical_names = {
        mbid: _select_canonical_name(names)
        for mbid, names in names_by_mbid.items()
    }
    return alias_to_mbid, canonical_names


def _parse_artist_credit(
    artist_credit: str,
    alias_to_mbid: dict[str, str],
) -> tuple[str, ...] | None:
    """Sépare un crédit sans MBID uniquement si tous ses artistes sont connus."""
    text = _normalize_text_value(artist_credit)
    aliases_by_first_character: dict[str, list[str]] = defaultdict(list)
    for alias in alias_to_mbid:
        if alias:
            aliases_by_first_character[alias[0]].append(alias)
    for aliases in aliases_by_first_character.values():
        aliases.sort(key=len, reverse=True)

    @lru_cache(maxsize=None)
    def parse_from(position: int) -> tuple[str, ...] | None:
        if position >= len(text):
            return None

        for alias in aliases_by_first_character.get(text[position], []):
            if not text.startswith(alias, position):
                continue

            end = position + len(alias)
            if end == len(text):
                return (alias,)

            separator = _ARTIST_SEPARATOR.match(text, end)
            if separator is None or separator.end() == end:
                continue

            remainder = parse_from(separator.end())
            if remainder:
                return (alias, *remainder)

        return None

    return parse_from(0)


def _resolved_credits(
    row: dict,
    alias_to_mbid: dict[str, str],
    canonical_names: dict[str, str],
) -> list[dict[str, str | None]]:
    """Retourne les artistes individuels d'une écoute et la source de l'identité."""
    if row["artist_credits"]:
        return [
            {
                "artist": canonical_names[credit["artist_mbid"]],
                "artist_mbid": credit["artist_mbid"],
                "artist_identity_source": "musicbrainz",
            }
            for credit in row["artist_credits"]
        ]

    aliases = _parse_artist_credit(row["artist_credit"], alias_to_mbid)
    if aliases:
        return [
            {
                "artist": canonical_names[alias_to_mbid[alias]],
                "artist_mbid": alias_to_mbid[alias],
                "artist_identity_source": "inferred_alias",
            }
            for alias in aliases
        ]

    return [
        {
            "artist": row["artist_credit"],
            "artist_mbid": None,
            "artist_identity_source": "text_fallback",
        }
    ]


def expand_artist_identities(df: pd.DataFrame) -> pd.DataFrame:
    """Produit une ligne par artiste crédité avec une identité stable."""
    if "artist_credits" not in df.columns:
        raise ValueError("La colonne artist_credits est indispensable")

    alias_to_mbid, canonical_names = _build_artist_catalog(df)
    expanded_rows: list[dict] = []

    for row in df.to_dict(orient="records"):
        credits = _resolved_credits(row, alias_to_mbid, canonical_names)
        credit_weight = 1 / len(credits)

        base_row = {key: value for key, value in row.items() if key != "artist_credits"}
        for credit in credits:
            artist_mbid = credit["artist_mbid"]
            artist_uid = (
                f"mbid:{artist_mbid}"
                if artist_mbid
                else f"text:{_normalize_text_value(credit['artist'])}"
            )
            expanded_rows.append(
                {
                    **base_row,
                    **credit,
                    "artist_uid": artist_uid,
                    "artist_credit_weight": credit_weight,
                }
            )

    expanded = pd.DataFrame(expanded_rows)

    # Les fallbacks texte partagent eux aussi un seul libellé canonique malgré
    # d'éventuelles différences de casse ou d'espacement.
    fallback_rows = expanded["artist_identity_source"].eq("text_fallback")
    if fallback_rows.any():
        fallback_names = (
            expanded.loc[fallback_rows]
            .groupby("artist_uid")["artist"]
            .agg(lambda values: _select_canonical_name(Counter(values)))
        )
        expanded.loc[fallback_rows, "artist"] = expanded.loc[
            fallback_rows, "artist_uid"
        ].map(fallback_names)

    # Deux artistes réellement distincts peuvent porter le même nom. Le suffixe
    # ne change que leur libellé d'affichage et évite que les notebooks actuels,
    # qui groupent encore sur `artist`, les fusionnent accidentellement.
    normalized_artist = _normalized_text(expanded["artist"])
    identities_per_name = expanded.groupby(normalized_artist)["artist_uid"].nunique()
    ambiguous_names = set(identities_per_name[identities_per_name > 1].index)
    ambiguous_rows = normalized_artist.isin(ambiguous_names)
    if ambiguous_rows.any():
        suffixes = expanded.loc[ambiguous_rows, "artist_mbid"].str[:8]
        suffixes = suffixes.fillna("unmapped")
        expanded.loc[ambiguous_rows, "artist"] = (
            expanded.loc[ambiguous_rows, "artist"] + " [" + suffixes + "]"
        )

    # Une même écoute peut être soumise avec deux crédits globaux différents.
    # Après résolution vers les MBID, ces doublons deviennent enfin visibles.
    expanded["__row_order"] = range(len(expanded))
    expanded["__source_priority"] = expanded["artist_identity_source"].map(
        {"musicbrainz": 0, "inferred_alias": 1, "text_fallback": 2}
    )
    expanded["__normalized_user"] = _normalized_text(expanded["user"])
    expanded["__normalized_track"] = _normalized_text(expanded["track"])
    interaction_key = [
        "__normalized_user",
        "artist_uid",
        "__normalized_track",
        "listened_at",
    ]
    expanded = expanded.sort_values("__source_priority", kind="stable")
    identity_duplicates = expanded.duplicated(subset=interaction_key, keep="first")
    identity_duplicate_count = int(identity_duplicates.sum())
    expanded = expanded.loc[~identity_duplicates].sort_values("__row_order")
    expanded = expanded.drop(
        columns=[
            "__row_order",
            "__source_priority",
            "__normalized_user",
            "__normalized_track",
        ]
    )

    ordered_columns = [
        "user",
        "artist_uid",
        "artist_mbid",
        "artist",
        "artist_credit",
        "artist_identity_source",
        "artist_credit_weight",
        "track",
        "release",
        "listened_at",
        "recording_msid",
    ]
    result = expanded[ordered_columns].reset_index(drop=True)
    result.attrs["identity_duplicates_removed"] = identity_duplicate_count
    return result


def build_user_quality_report(
    listens: pd.DataFrame,
    *,
    min_distinct_artists: int = DEFAULT_MIN_DISTINCT_ARTISTS,
    max_dominant_artist_share: float = DEFAULT_MAX_DOMINANT_ARTIST_SHARE,
    min_history_span: str | pd.Timedelta = DEFAULT_MIN_HISTORY_SPAN,
) -> pd.DataFrame:
    """Mesure la qualité de chaque profil et indique les motifs d'exclusion."""
    required_columns = [
        "user",
        "artist_uid",
        "artist",
        "track",
        "listened_at",
        "recording_msid",
    ]
    missing_columns = set(required_columns).difference(listens.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colonnes indispensables absentes : {missing}")
    if min_distinct_artists < 1:
        raise ValueError("min_distinct_artists doit être positif")
    if not 0 < max_dominant_artist_share <= 1:
        raise ValueError("max_dominant_artist_share doit être compris entre 0 et 1")

    minimum_span = pd.Timedelta(min_history_span)
    if minimum_span <= pd.Timedelta(0):
        raise ValueError("min_history_span doit être positif")

    working = listens.copy()
    fallback_event_id = (
        working["listened_at"].astype("string")
        + "|"
        + _normalized_text(working["track"])
    )
    working["__event_id"] = (
        working["recording_msid"].astype("string").fillna(fallback_event_id)
    )

    quality = (
        working.groupby("user")
        .agg(
            interaction_rows=("artist_uid", "size"),
            n_listens=("__event_id", "nunique"),
            n_distinct_artists=("artist_uid", "nunique"),
            history_start=("listened_at", "min"),
            history_end=("listened_at", "max"),
        )
        .reset_index()
    )
    quality["history_span"] = quality["history_end"] - quality["history_start"]
    quality["history_span_hours"] = (
        quality["history_span"].dt.total_seconds() / 3600
    )

    artist_names = (
        working[["artist_uid", "artist"]]
        .drop_duplicates("artist_uid")
        .set_index("artist_uid")["artist"]
    )
    artist_counts = (
        working.groupby(["user", "artist_uid"])["__event_id"]
        .nunique()
        .rename("dominant_artist_listens")
        .reset_index()
        .sort_values(
            ["user", "dominant_artist_listens", "artist_uid"],
            ascending=[True, False, True],
            kind="stable",
        )
    )
    dominant = artist_counts.drop_duplicates("user").rename(
        columns={"artist_uid": "dominant_artist_uid"}
    )
    dominant["dominant_artist"] = dominant["dominant_artist_uid"].map(
        artist_names
    )
    quality = quality.merge(dominant, on="user", validate="one_to_one")
    quality["dominant_artist_share"] = (
        quality["dominant_artist_listens"] / quality["n_listens"]
    )

    quality["too_few_artists"] = (
        quality["n_distinct_artists"] < min_distinct_artists
    )
    quality["dominant_artist_over_limit"] = (
        quality["dominant_artist_share"] > max_dominant_artist_share
    )
    quality["history_too_short"] = quality["history_span"] < minimum_span
    quality["is_usable"] = ~quality[
        [
            "too_few_artists",
            "dominant_artist_over_limit",
            "history_too_short",
        ]
    ].any(axis=1)

    def exclusion_reasons(row: pd.Series) -> str:
        reasons: list[str] = []
        if row["too_few_artists"]:
            reasons.append("too_few_artists")
        if row["dominant_artist_over_limit"]:
            reasons.append("dominant_artist_over_limit")
        if row["history_too_short"]:
            reasons.append("history_too_short")
        return ";".join(reasons)

    quality["exclusion_reasons"] = quality.apply(exclusion_reasons, axis=1)
    return quality.sort_values("user", kind="stable").reset_index(drop=True)


def filter_usable_users(
    listens: pd.DataFrame,
    *,
    min_distinct_artists: int = DEFAULT_MIN_DISTINCT_ARTISTS,
    max_dominant_artist_share: float = DEFAULT_MAX_DOMINANT_ARTIST_SHARE,
    min_history_span: str | pd.Timedelta = DEFAULT_MIN_HISTORY_SPAN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conserve les profils exploitables et retourne aussi le rapport complet."""
    quality_report = build_user_quality_report(
        listens,
        min_distinct_artists=min_distinct_artists,
        max_dominant_artist_share=max_dominant_artist_share,
        min_history_span=min_history_span,
    )
    usable_users = set(
        quality_report.loc[quality_report["is_usable"], "user"]
    )
    filtered = listens[listens["user"].isin(usable_users)].copy()
    filtered.attrs.update(listens.attrs)
    return filtered.reset_index(drop=True), quality_report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_data_fingerprint(files: list[Path], raw_dir: Path) -> str:
    """Calcule une empreinte sensible aux noms et au contenu des fichiers bruts."""
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(raw_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _copy_immutable_version(source: Path, destination: Path) -> None:
    """Crée un snapshot sans jamais remplacer silencieusement une autre version."""
    if destination.exists():
        if _sha256_file(source) != _sha256_file(destination):
            raise RuntimeError(
                f"Le snapshot {destination} existe avec un contenu différent. "
                "Incrémente CLEANING_PIPELINE_VERSION."
            )
        return
    destination.write_bytes(source.read_bytes())


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def build_processed_dataset(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    *,
    min_distinct_artists: int = DEFAULT_MIN_DISTINCT_ARTISTS,
    max_dominant_artist_share: float = DEFAULT_MAX_DOMINANT_ARTIST_SHARE,
    min_history_span: str | pd.Timedelta = DEFAULT_MIN_HISTORY_SPAN,
) -> CleaningResult:
    """Construit le Parquet nettoyé et retourne son bilan."""
    raw_files = sorted(raw_dir.glob(RAW_FILE_PATTERN))
    raw_fingerprint = _raw_data_fingerprint(raw_files, raw_dir)
    minimum_span = pd.Timedelta(min_history_span)
    cleaning_config = {
        "min_distinct_artists": min_distinct_artists,
        "max_dominant_artist_share": max_dominant_artist_share,
        "min_history_span": str(minimum_span),
    }
    version_payload = json.dumps(
        {
            "pipeline_version": CLEANING_PIPELINE_VERSION,
            "raw_data_sha256": raw_fingerprint,
            "cleaning_config": cleaning_config,
        },
        sort_keys=True,
    ).encode("utf-8")
    version_hash = hashlib.sha256(version_payload).hexdigest()[:12]
    dataset_version = f"clean-v{CLEANING_PIPELINE_VERSION}-{version_hash}"

    raw_df = load_raw_listens(raw_dir)
    cleaned_df, raw_duplicate_count = clean_listens(raw_df)
    expanded_df = expand_artist_identities(cleaned_df)
    identity_duplicate_count = int(
        expanded_df.attrs.get("identity_duplicates_removed", 0)
    )
    duplicate_count = raw_duplicate_count + identity_duplicate_count
    filtered_df, quality_report = filter_usable_users(
        expanded_df,
        min_distinct_artists=min_distinct_artists,
        max_dominant_artist_share=max_dominant_artist_share,
        min_history_span=minimum_span,
    )
    filtered_df["dataset_version"] = dataset_version
    quality_report["dataset_version"] = dataset_version

    processed_dir.mkdir(parents=True, exist_ok=True)
    versions_dir = processed_dir / VERSIONS_DIRECTORY
    versions_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / OUTPUT_FILENAME
    quality_report_path = processed_dir / QUALITY_REPORT_FILENAME
    cleaning_report_path = processed_dir / CLEANING_REPORT_FILENAME
    versioned_output_path = versions_dir / f"{dataset_version}.parquet"
    versioned_quality_path = versions_dir / f"{dataset_version}.user_quality.parquet"
    versioned_report_path = versions_dir / f"{dataset_version}.cleaning_report.json"
    filtered_df.to_parquet(output_path, index=False)
    quality_report.to_parquet(quality_report_path, index=False)
    _copy_immutable_version(output_path, versioned_output_path)
    _copy_immutable_version(quality_report_path, versioned_quality_path)

    identity_counts = {
        str(source): int(count)
        for source, count in filtered_df["artist_identity_source"].value_counts().items()
    }
    filter_counts = {
        "users_before": int(quality_report.shape[0]),
        "users_after": int(quality_report["is_usable"].sum()),
        "users_excluded": int((~quality_report["is_usable"]).sum()),
        "rows_excluded": int(len(expanded_df) - len(filtered_df)),
    }
    excluded_quality = quality_report.loc[~quality_report["is_usable"]]
    exact_reason_summary = [
        {
            "reason": str(reason),
            "users_removed": int(len(group)),
            "rows_removed": int(group["interaction_rows"].sum()),
        }
        for reason, group in excluded_quality.groupby(
            "exclusion_reasons", sort=True
        )
    ]
    flag_summary = {
        flag: {
            "users": int(excluded_quality[flag].sum()),
            "rows": int(
                excluded_quality.loc[
                    excluded_quality[flag], "interaction_rows"
                ].sum()
            ),
        }
        for flag in (
            "too_few_artists",
            "dominant_artist_over_limit",
            "history_too_short",
        )
    }

    project_root = processed_dir.parent.parent
    expanded_before_identity_dedup = len(expanded_df) + identity_duplicate_count
    report = {
        "dataset_version": dataset_version,
        "pipeline_version": CLEANING_PIPELINE_VERSION,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": {
            "raw_directory": _relative_path(raw_dir, project_root),
            "file_pattern": RAW_FILE_PATTERN,
            "file_count": len(raw_files),
            "raw_data_sha256": raw_fingerprint,
        },
        "cleaning_config": cleaning_config,
        "rows": {
            "raw_loaded": int(len(raw_df)),
            "raw_duplicates_removed": raw_duplicate_count,
            "after_raw_deduplication": int(len(cleaned_df)),
            "after_artist_expansion_before_identity_deduplication": int(
                expanded_before_identity_dedup
            ),
            "identity_duplicates_removed": identity_duplicate_count,
            "after_identity_deduplication": int(len(expanded_df)),
            "unusable_user_rows_removed": filter_counts["rows_excluded"],
            "final": int(len(filtered_df)),
        },
        "users": {
            "raw": int(raw_df["user"].nunique()),
            "before_quality_filter": filter_counts["users_before"],
            "after_quality_filter": filter_counts["users_after"],
            "excluded": filter_counts["users_excluded"],
        },
        "identity_resolution_final": identity_counts,
        "user_filter": {
            "exact_reason_combinations": exact_reason_summary,
            "non_exclusive_flags": flag_summary,
        },
        "outputs": {
            "current_dataset": _relative_path(output_path, project_root),
            "versioned_dataset": _relative_path(
                versioned_output_path, project_root
            ),
            "quality_report": _relative_path(quality_report_path, project_root),
            "versioned_quality_report": _relative_path(
                versioned_quality_path, project_root
            ),
            "dataset_sha256": _sha256_file(output_path),
            "quality_report_sha256": _sha256_file(quality_report_path),
        },
    }
    report_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    cleaning_report_path.write_text(report_text, encoding="utf-8")
    if not versioned_report_path.exists():
        versioned_report_path.write_text(report_text, encoding="utf-8")

    return CleaningResult(
        output_path=output_path,
        versioned_output_path=versioned_output_path,
        quality_report_path=quality_report_path,
        cleaning_report_path=cleaning_report_path,
        dataset_version=dataset_version,
        row_count=len(filtered_df),
        duplicate_count=duplicate_count,
        identity_counts=identity_counts,
        filter_counts=filter_counts,
    )


def main() -> None:
    result = build_processed_dataset()
    print(f"Version : {result.dataset_version}")
    print(f"{result.duplicate_count} doublon(s) supprimé(s)")
    print(
        f"{result.row_count} interaction(s) utilisateur-artiste conservée(s)"
    )
    print(
        "Identités : "
        + ", ".join(
            f"{source}={count}"
            for source, count in result.identity_counts.items()
        )
    )
    print(
        f"{result.filter_counts['users_excluded']} utilisateur(s) exclu(s), "
        f"{result.filter_counts['users_after']} conservé(s)"
    )
    print(f"Rapport qualité : {result.quality_report_path}")
    print(f"Journal : {result.cleaning_report_path}")
    print(f"Fichier courant : {result.output_path}")
    print(f"Snapshot versionné : {result.versioned_output_path}")


if __name__ == "__main__":
    main()
