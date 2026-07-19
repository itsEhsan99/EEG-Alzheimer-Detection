"""Tests for dataset-building helpers."""

import pytest

from src.dataset import subject_id


def test_subject_id_extraction():
    assert subject_id("sub-001_task-eyesclosed_eeg.set") == "sub-001"
    assert subject_id("DATA/HC/sub-046_task-eyesclosed_eeg.set") == "sub-046"


def test_subject_id_missing_raises():
    with pytest.raises(ValueError):
        subject_id("random_file.set")