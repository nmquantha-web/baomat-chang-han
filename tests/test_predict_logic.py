import numpy as np
import pandas as pd

from predict import prepare_feature_frame, create_inference_sequences


def test_prepare_feature_frame_reorders_and_fills_missing_values():
    df = pd.DataFrame({
        "B": [1.0, 2.0],
        "C": [3.0, np.nan],
    })
    metadata = {
        "feature_names": ["A", "B", "C"],
        "medians": {"A": 0.5, "B": 1.5, "C": 3.0},
    }

    out = prepare_feature_frame(df, metadata)

    assert list(out.columns) == ["A", "B", "C"]
    assert out["A"].isna().sum() == 0
    assert out["A"].tolist() == [0.5, 0.5]
    assert out["B"].tolist() == [1.0, 2.0]
    assert out["C"].tolist() == [3.0, 3.0]


def test_create_inference_sequences_keeps_one_sequence_per_row():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ], dtype=np.float32)

    seqs = create_inference_sequences(X, window_size=3)

    assert seqs.shape[0] == X.shape[0]
    assert seqs.shape[1:] == (3, 2)
    np.testing.assert_allclose(seqs[0, :, :], np.array([[0, 0], [1, 2], [1, 2]], dtype=np.float32))
