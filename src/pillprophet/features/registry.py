"""Feature registry: tracks source, time availability, and leakage status.

The registry is built from the structured and text YAML configs and provides
a single place to query which features exist, where they come from, and
whether they are safe to use at a given timepoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pillprophet.snapshots.availability import TIMEPOINTS, check_leakage
from pillprophet.utils.config import load_config

logger = logging.getLogger("pillprophet")


@dataclass
class FeatureEntry:
    """Metadata for a single feature."""

    name: str
    source_column: str
    feature_type: str  # "categorical", "numeric", "derived", "tfidf"
    time_availability: str
    extra: dict = field(default_factory=dict)


class FeatureRegistry:
    """In-memory registry of all features with leakage validation."""

    def __init__(self) -> None:
        self._entries: dict[str, FeatureEntry] = {}

    # ── Building the registry ───────────────────────────────────────────

    def register(self, entry: FeatureEntry) -> None:
        self._entries[entry.name] = entry

    @classmethod
    def from_configs(
        cls,
        structured_config_path: str | Path,
        text_config_path: str | Path,
    ) -> FeatureRegistry:
        """Build a registry from the structured + text YAML configs."""
        registry = cls()

        # Structured features.
        scfg = load_config(structured_config_path)
        for feat in scfg.get("categorical", []):
            registry.register(FeatureEntry(
                name=feat["name"],
                source_column=feat["source_column"],
                feature_type="categorical",
                time_availability=feat["time_availability"],
            ))
        for feat in scfg.get("numeric", []):
            registry.register(FeatureEntry(
                name=feat["name"],
                source_column=feat["source_column"],
                feature_type="numeric",
                time_availability=feat["time_availability"],
                extra={k: v for k, v in feat.items()
                       if k not in ("name", "source_column", "time_availability")},
            ))
        for feat in scfg.get("derived", []):
            registry.register(FeatureEntry(
                name=feat["name"],
                source_column=feat["source_column"],
                feature_type="derived",
                time_availability=feat["time_availability"],
                extra={k: v for k, v in feat.items()
                       if k not in ("name", "source_column", "time_availability")},
            ))

        # Text features — one entry per text field, plus "tfidf" block.
        tcfg = load_config(text_config_path)
        for tf in tcfg.get("text_fields", []):
            registry.register(FeatureEntry(
                name=f"text__{tf['name']}",
                source_column=tf["source_column"],
                feature_type="text_source",
                time_availability=tf["time_availability"],
            ))
        # Register the TF-IDF output as a single virtual entry.
        registry.register(FeatureEntry(
            name="tfidf_matrix",
            source_column="(combined text)",
            feature_type="tfidf",
            time_availability="T0",
            extra=tcfg.get("tfidf", {}),
        ))

        logger.info("Feature registry: %d entries loaded.", len(registry))
        return registry

    # ── Query API ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries.values())

    def get(self, name: str) -> FeatureEntry | None:
        return self._entries.get(name)

    def names(self) -> list[str]:
        return list(self._entries.keys())

    def by_type(self, feature_type: str) -> list[FeatureEntry]:
        return [e for e in self._entries.values() if e.feature_type == feature_type]

    # ── Leakage validation ──────────────────────────────────────────────

    def validate_for_timepoint(self, timepoint: str) -> list[str]:
        """Return names of features whose *source columns* would leak at *timepoint*.

        This checks the underlying source column against the field-level
        availability registry in ``snapshots.availability``.
        """
        if timepoint not in TIMEPOINTS:
            raise ValueError(f"Unknown timepoint {timepoint!r}")

        source_cols = [e.source_column for e in self._entries.values()
                       if e.source_column != "(combined text)"]
        leaking_cols = set(check_leakage(source_cols, timepoint))

        return [e.name for e in self._entries.values()
                if e.source_column in leaking_cols]

    def safe_features(self, timepoint: str) -> list[str]:
        """Return feature names that are safe to use at *timepoint*."""
        unsafe = set(self.validate_for_timepoint(timepoint))
        return [n for n in self._entries if n not in unsafe]


# ── Convenience loaders ─────────────────────────────────────────────────────

def load_feature_registry(
    structured_config_path: str | Path,
    text_config_path: str | Path,
) -> FeatureRegistry:
    """Load the feature registry from config files."""
    return FeatureRegistry.from_configs(structured_config_path, text_config_path)
