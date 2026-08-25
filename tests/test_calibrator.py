import pytest

from cad.calibrator import CADCalibrator


class ProbeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1] if "up" in text.lower() else [2]


def test_entity_and_date_scales_reach_profiled_alpha():
    calibrator = CADCalibrator(
        model=object(),
        tokenizer=ProbeTokenizer(),
        device="cpu",
        logit_gap_profile={
            "alpha_max": 3.0,
            "alpha_min": 0.0,
            "entropy_oos_std": 0.08,
            "entropy_in_sample_std": 0.10,
        },
    )
    calibrator._entity_date_variance["NVDA"] = 0.08
    calibrator._entity_mean_entropy["NVDA"] = 0.80
    calibrator._probe = lambda prompt: (0.70, 0.8, 0.2, 2.0, 1.0)

    result = calibrator.calibrate_alpha("NVDA", date="2018-06-29")

    assert result.alpha == pytest.approx(3.0)
    assert result.delta_temporal == pytest.approx(0.10)
    assert result.entity_date_var == pytest.approx(0.08)


def test_no_excess_date_confidence_means_no_penalty():
    calibrator = CADCalibrator(object(), ProbeTokenizer(), device="cpu")
    calibrator._entity_date_variance["NVDA"] = 0.20
    calibrator._entity_mean_entropy["NVDA"] = 0.60
    calibrator._probe = lambda prompt: (0.70, 0.5, 0.5, 1.0, 1.0)

    assert calibrator.calibrate_alpha("NVDA", date="2025-06-30").alpha == 0.0
