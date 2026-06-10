"""Per-stage artifact store for the Orchestrator (Req 10.3, 10.4).

Each pipeline stage produces an artifact that the next stage consumes
(Parsed_Document → Normalized_Document → Chunk set → Embedding set → stored
records). The Orchestrator persists every stage's output here and records a
pointer to it (``StageState.artifactRef``) so that:

* a transition is fully observable — the stored artifact backs the recorded
  ``artifactRef`` (Req 10.6), and
* a resumed job can re-enter at the failing stage while reusing the preserved
  upstream artifacts (Req 10.3, 10.4) — wired up in task 13.2.

The store is a deliberately small in-memory key/value map keyed by an opaque
:data:`ArtifactRef` string. The Orchestrator owns ref construction; callers
treat the ref as opaque. A persistent adapter could later implement the same
surface (the normalized-document artifact is stored in its durable serialized
byte form precisely so a durable backend is a drop-in replacement).
"""

from __future__ import annotations

from typing import Dict

# An opaque pointer to a persisted stage artifact (recorded on StageState).
ArtifactRef = str


class UnknownArtifactError(KeyError):
    """Raised when an artifact ref is not present in the store."""

    def __init__(self, ref: ArtifactRef) -> None:
        self.ref = ref
        super().__init__(f"no artifact stored for ref {ref!r}")


class ArtifactStore:
    """An in-memory store of per-stage artifacts keyed by :data:`ArtifactRef`.

    Invariants:

    * ``put`` returns the ref it stored under; storing again under the same ref
      replaces the prior artifact (a re-run of a stage overwrites its output).
    * ``get`` raises :class:`UnknownArtifactError` for an absent ref so a broken
      ``artifactRef`` surfaces loudly rather than silently yielding ``None``.
    """

    def __init__(self) -> None:
        self._artifacts: Dict[ArtifactRef, object] = {}

    def put(self, ref: ArtifactRef, artifact: object) -> ArtifactRef:
        """Persist ``artifact`` under ``ref`` and return ``ref``."""
        if not isinstance(ref, str) or not ref:
            raise TypeError("artifact ref must be a non-empty str")
        self._artifacts[ref] = artifact
        return ref

    def get(self, ref: ArtifactRef) -> object:
        """Return the artifact stored under ``ref`` or raise."""
        try:
            return self._artifacts[ref]
        except KeyError:
            raise UnknownArtifactError(ref) from None

    def has(self, ref: ArtifactRef) -> bool:
        """Return whether an artifact is stored under ``ref``."""
        return ref in self._artifacts

    def __contains__(self, ref: object) -> bool:
        return ref in self._artifacts

    def __len__(self) -> int:
        return len(self._artifacts)
