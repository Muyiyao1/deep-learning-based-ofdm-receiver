from __future__ import annotations

import numpy as np
import pandas as pd

from src.plots import LOG_FLOOR, _ber_values, _log_ci_errors


def test_log_ci_omits_only_a_nonpositive_lower_arm() -> None:
    values = np.array([1.2e-3, 5.0e-3])
    ci95 = np.array([1.3e-3, 1.0e-3])

    plotted, errors, crosses_zero = _log_ci_errors(values, ci95)

    np.testing.assert_allclose(plotted, values)
    np.testing.assert_allclose(errors[0], [0.0, 1.0e-3])
    np.testing.assert_allclose(errors[1], ci95)
    np.testing.assert_array_equal(crosses_zero, [True, False])


def test_log_ci_applies_a_floor_only_to_nonpositive_means() -> None:
    plotted, errors, crosses_zero = _log_ci_errors(np.array([0.0]), np.array([0.1]))

    np.testing.assert_allclose(plotted, [LOG_FLOOR])
    np.testing.assert_allclose(errors, [[0.0], [0.1]])
    np.testing.assert_array_equal(crosses_zero, [True])


def test_ber_values_falls_back_to_the_seed_mean_when_pooled_value_is_missing() -> None:
    group = pd.DataFrame({"ber_plot": [0.01, np.nan], "ber_mean": [0.02, 0.03]})

    np.testing.assert_allclose(_ber_values(group), [0.01, 0.03])
