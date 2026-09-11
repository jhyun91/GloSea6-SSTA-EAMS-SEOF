import numpy as np

from eams_seof.verification import acc, bh_fdr, msss


def test_uncentered_acc_uses_zero_anomaly_reference():
    forecast = np.array([1.0, 2.0, 3.0])
    observed = np.array([2.0, 2.0, 4.0])
    expected = np.sum(forecast * observed) / np.sqrt(
        np.sum(forecast ** 2) * np.sum(observed ** 2)
    )
    assert np.isclose(acc(forecast, observed), expected)


def test_msss_uses_zero_anomaly_climatology():
    observed = np.array([1.0, -1.0, 2.0, -2.0])
    perfect = observed.copy()
    climatology = np.zeros_like(observed)
    assert np.isclose(msss(perfect, observed), 1.0)
    assert np.isclose(msss(climatology, observed), 0.0)


def test_bh_fdr_q_point_one():
    p = np.array([0.005, 0.02, 0.20, np.nan])
    reject, adjusted = bh_fdr(p, alpha=0.10)
    assert reject.tolist() == [True, True, False, False]
    assert np.isnan(adjusted[-1])


def test_null_based_skill_pvalues_detect_perfect_skill():
    from eams_seof.verification import null_pvalue

    observed = np.array([1.0, -1.0, 2.0, -2.0, 1.5, -1.5, 2.5, -2.5, 3.0, -3.0, 0.5, -0.5])
    forecast = observed.copy()

    assert null_pvalue(forecast, observed, metric="ACC", n_null=2000, min_samples=10, seed=1) < 0.05
    assert null_pvalue(forecast, observed, metric="MSSS", n_null=2000, min_samples=10, seed=1) < 0.05
