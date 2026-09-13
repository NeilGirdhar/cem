from cem.demos.supervised.missingness import summarize_missingness_losses


def test_summarize_missingness_losses_reports_means_and_deviations() -> None:
    losses = {
        "npn": [[1.0, 3.0], [2.0, 6.0]],
        "mask-mlp": [[2.0, 4.0], [4.0, 8.0]],
    }

    result = summarize_missingness_losses(losses)

    assert result["npn"] == [2.0, 4.0]
    assert result["mask-mlp"] == [3.0, 6.0]
    assert result["bar errors"] == {
        "npn": [1.0, 2.0],
        "mask-mlp": [1.0, 2.0],
    }
    assert result["seed losses"] == losses
