from tools.compare_predictions import compare_group, scalar


def test_zero_sign_flips_only_for_thresholded_groups():
    rows, flips = compare_group(
        {"a": [-0.1], "b": [0.2]},
        {"a": [0.1], "b": [0.3]},
        track_zero_flip=True,
    )
    assert flips == 1
    assert rows["a"]["sign_flip_at_zero"] is True
    assert rows["b"]["sign_flip_at_zero"] is False


def test_priority_rows_do_not_report_zero_sign_flip():
    rows, flips = compare_group(
        {"none": [-1.0], "high": [2.0]},
        {"none": [0.1], "high": [1.5]},
        track_zero_flip=False,
    )
    assert flips == 0
    assert "sign_flip_at_zero" not in rows["none"]
    assert scalar([1.5]) == 1.5


def test_compare_group_marks_missing_outputs():
    rows, flips = compare_group(
        {"a": [0.2], "missing": [0.1]},
        {"a": [0.3]},
        track_zero_flip=True,
    )
    assert flips == 0
    assert rows["missing"] == {"missing": True}
