"""Bounded undo/redo history of screw-plan snapshots."""

import copy
from typing import Hashable, List, Optional, Tuple

from src.models.screw import Screw

Step = Tuple[str, List[Screw]]


class ScrewHistory:
    """Undo and redo stacks of whole screw lists.

    Each step stores a deep copy of the list as it was before one edit, with a
    short label for the menu and status bar ("delete screw"). The caller
    restores a returned list through its normal rebuild path.
    """

    def __init__(self, limit: int = 50):
        self._limit = int(limit)
        self._undo: List[Step] = []
        self._redo: List[Step] = []
        self._coalesce_key: Optional[Hashable] = None

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_label(self) -> Optional[str]:
        return self._undo[-1][0] if self._undo else None

    @property
    def redo_label(self) -> Optional[str]:
        return self._redo[-1][0] if self._redo else None

    def push(
        self,
        label: str,
        before: List[Screw],
        coalesce_key: Optional[Hashable] = None,
    ) -> None:
        """Record the list as it was before an edit; drops the redo stack.

        Consecutive pushes with the same non-None ``coalesce_key`` (a burst of
        spin-box steps on one screw) keep only the first snapshot.
        """
        self._redo.clear()
        if coalesce_key is not None and coalesce_key == self._coalesce_key:
            return
        self._coalesce_key = coalesce_key
        self._undo.append((label, copy.deepcopy(list(before))))
        del self._undo[: -self._limit]

    def undo(self, current: List[Screw]) -> Optional[Step]:
        """Pop the last step; ``current`` becomes the matching redo step."""
        return self._move(self._undo, self._redo, current)

    def redo(self, current: List[Screw]) -> Optional[Step]:
        """Pop the last undone step; ``current`` goes back on the undo stack."""
        return self._move(self._redo, self._undo, current)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._coalesce_key = None

    def _move(self, source, target, current) -> Optional[Step]:
        self._coalesce_key = None
        if not source:
            return None
        label, screws = source.pop()
        target.append((label, copy.deepcopy(list(current))))
        del target[: -self._limit]
        return label, screws
