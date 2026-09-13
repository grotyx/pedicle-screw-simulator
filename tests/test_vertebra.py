"""Direct tests for the untested vertebra data models.

Regression scope: src/core/vertebra.py dataclass validation. Post-init
normalises array-likes to float64 numpy arrays and rejects wrong shapes.
Uses synthetic in-code arrays only.
"""

import numpy as np
import pytest

from src.core.vertebra import PedicleAnalysisResult, Vertebra, aiming_endplate_normal


def _vertebra(**overrides):
    kwargs = {
        "label": 28,
        "name": "L4",
        "centroid_lps": [1.0, 2.0, 3.0],
        "bounding_box": ([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
        "volume_mm3": 30000.0,
        "mask_indices": np.zeros((4, 3), dtype=int),
    }
    kwargs.update(overrides)
    return Vertebra(**kwargs)


def test_vertebra_normalises_arrays_to_float64():
    vertebra = _vertebra()

    assert vertebra.centroid_lps.dtype == np.float64
    assert vertebra.bounding_box[0].dtype == np.float64
    assert vertebra.bounding_box[1].dtype == np.float64
    np.testing.assert_allclose(vertebra.centroid_lps, [1.0, 2.0, 3.0])


def test_vertebra_rejects_bad_centroid_shape():
    with pytest.raises(ValueError, match=r"centroid_lps must have shape \(3,\)"):
        _vertebra(centroid_lps=[1.0, 2.0])


def test_vertebra_rejects_bad_bounding_box_shape():
    with pytest.raises(ValueError, match="bounding_box corners"):
        _vertebra(bounding_box=([0.0, 0.0], [1.0, 1.0]))


def test_vertebra_rejects_bad_mask_indices_shape():
    with pytest.raises(ValueError, match=r"mask_indices must have shape \(N, 3\)"):
        _vertebra(mask_indices=np.zeros((4, 2), dtype=int))
    with pytest.raises(ValueError, match=r"mask_indices must have shape \(N, 3\)"):
        _vertebra(mask_indices=np.zeros(4, dtype=int))


def test_aiming_endplate_normal_unresolved_returns_own_fit():
    own = np.array([0.0, 0.0, 1.0])
    analysis = PedicleAnalysisResult(
        vertebra=_vertebra(), upper_endplate_normal=own
    )

    assert aiming_endplate_normal(analysis) is own


def test_aiming_endplate_normal_resolved_returns_reference():
    own = np.array([0.0, 0.0, 1.0])
    reference = np.array([0.1, 0.0, 1.0])
    analysis = PedicleAnalysisResult(
        vertebra=_vertebra(),
        upper_endplate_normal=own,
        endplate_reference="neighbours",
        reference_endplate_normal=reference,
        endplate_reference_levels=("L3", "L5"),
    )

    assert aiming_endplate_normal(analysis) is reference


def test_aiming_endplate_normal_resolved_none_returns_none():
    analysis = PedicleAnalysisResult(
        vertebra=_vertebra(),
        upper_endplate_normal=np.array([0.0, 0.0, 1.0]),
        endplate_reference="none",
        reference_endplate_normal=None,
    )

    assert aiming_endplate_normal(analysis) is None
