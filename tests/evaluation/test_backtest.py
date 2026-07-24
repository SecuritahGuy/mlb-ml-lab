"""Tests for the backtesting module."""

from __future__ import annotations

from datetime import date

import pytest

from mlb_ml_lab.evaluation.backtest import (
    GamePrediction,
    BetResult,
    calibration_buckets,
    expected_calibration_error,
    max_drawdown,
    simulate_bets,
    walk_forward_predict,
    apply_calibrators,
    fit_season_calibrators,
    load_calibrators,
    save_calibrators,
)


# ---------------------------------------------------------------------------
# walk_forward_predict
# ---------------------------------------------------------------------------


def _feature_row(
    player_id: int,
    game_pk: int,
    d: str,
    feat_val: float = 0.5,
) -> dict:
    return {
        "player_id": player_id,
        "game_pk": game_pk,
        "date": date.fromisoformat(d),
        "rolling_avg": feat_val,
    }


def _target_row(
    player_id: int,
    game_pk: int,
    d: str,
    hits: int = 0,
) -> dict:
    return {
        "player_id": player_id,
        "game_pk": game_pk,
        "date": date.fromisoformat(d),
        "hits": hits,
        "target_0.5": 1 if hits > 0 else 0,
        "target_1.5": 1 if hits > 1 else 0,
    }


class TestWalkForwardPredict:
    def _make_dates(self, n: int) -> list[date]:
        """Generate *n* unique dates starting from 2025-04-01."""
        from datetime import timedelta

        start = date(2025, 4, 1)
        return [start + timedelta(days=i) for i in range(n)]

    def test_returns_game_predictions(self):
        n = 31
        # Mix positive and negative examples so the model can train
        feat = [
            _feature_row(
                1, 100 + i, d.isoformat(), feat_val=0.5 + (0.1 if i < 15 else -0.1)
            )
            for i, d in enumerate(self._make_dates(n))
        ]
        tgt = [
            _target_row(1, 100 + i, d.isoformat(), hits=1 if i < 15 else 0)
            for i, d in enumerate(self._make_dates(n))
        ]
        preds = walk_forward_predict(
            feat,
            tgt,
            target_col="target_0.5",
            model_type="lr",
            n_splits=1,
        )
        assert len(preds) >= 1
        assert isinstance(preds[0], GamePrediction)

    def test_empty_inputs(self):
        assert walk_forward_predict([], []) == []

    def test_missing_player_id_raises(self):
        feat = [{"game_pk": 100, "date": date(2025, 4, 1), "rolling_avg": 0.5}]
        tgt = [_target_row(1, 100, "2025-04-01")]
        with pytest.raises(KeyError):
            walk_forward_predict(feat, tgt)


# ---------------------------------------------------------------------------
# simulate_bets
# ---------------------------------------------------------------------------


def _prediction(prob: float, actual: int, d: str = "2025-04-01") -> GamePrediction:
    return GamePrediction(
        date=date.fromisoformat(d),
        player_id=1,
        game_pk=100,
        predicted_prob=prob,
        actual=actual,
        hits=actual,
        target_col="target_0.5",
    )


class TestSimulateBets:
    def test_all_wins(self):
        preds = [_prediction(0.9, 1) for _ in range(10)]
        result = simulate_bets(preds, decimal_odds=2.0, stake_per_bet=1.0)
        assert result.total_bets == 10
        assert result.wins == 10
        assert result.losses == 0
        assert result.win_rate == 1.0
        assert result.total_profit == 10.0  # $1 profit per bet at 2.0 odds
        assert result.roi == 1.0  # 100% ROI

    def test_all_losses(self):
        preds = [_prediction(0.9, 0) for _ in range(10)]
        result = simulate_bets(preds, decimal_odds=2.0, stake_per_bet=1.0)
        assert result.total_bets == 10
        assert result.wins == 0
        assert result.losses == 10
        assert result.win_rate == 0.0
        assert result.total_profit == -10.0
        assert result.roi == -1.0

    def test_mixed_results(self):
        preds = [_prediction(0.9, 1), _prediction(0.9, 0)]
        result = simulate_bets(preds, decimal_odds=2.0, stake_per_bet=1.0)
        assert result.total_bets == 2
        assert result.wins == 1
        assert result.losses == 1
        assert result.win_rate == 0.5
        assert result.total_profit == 0.0  # +$1 win, -$1 loss

    def test_empty_predictions(self):
        result = simulate_bets([], decimal_odds=2.0)
        assert result.total_bets == 0
        assert result.total_profit == 0.0
        assert result.roi == 0.0

    def test_min_prob_threshold_filters(self):
        preds = [
            _prediction(0.5, 1),
            _prediction(0.7, 1),
            _prediction(0.9, 0),
        ]
        result = simulate_bets(preds, decimal_odds=2.0, min_prob=0.75)
        assert result.total_bets == 1  # only the 0.9 passes
        assert result.wins == 0  # the 0.9 was a loss

    def test_default_min_prob_is_breakeven(self):
        preds = [
            _prediction(0.4, 1),
            _prediction(0.6, 1),
        ]
        # breakeven for 2.0 odds = 0.5
        result = simulate_bets(preds, decimal_odds=2.0)
        assert result.total_bets == 1  # only 0.6 passes
        assert result.threshold == 0.5

    def test_custom_stake(self):
        preds = [_prediction(0.9, 1)]
        result = simulate_bets(preds, decimal_odds=2.0, stake_per_bet=5.0)
        assert result.total_stake == 5.0
        assert result.total_profit == 5.0  # $5 profit at 2.0

    def test_max_drawdown_nonzero_on_loss_streak(self):
        # Alternating losses then wins — should have a drawdown
        preds = [
            _prediction(0.9, 0, d="2025-04-01"),
            _prediction(0.9, 0, d="2025-04-02"),
            _prediction(0.9, 1, d="2025-04-03"),
        ]
        result = simulate_bets(preds, decimal_odds=2.0, stake_per_bet=1.0)
        assert result.max_drawdown > 0.0

    def test_zero_drawdown_on_all_wins(self):
        preds = [_prediction(0.9, 1, d=f"2025-04-{i:02d}") for i in range(1, 6)]
        result = simulate_bets(preds, decimal_odds=2.0, stake_per_bet=1.0)
        assert result.max_drawdown == 0.0

    def test_predicted_prob_mean(self):
        preds = [
            _prediction(0.6, 1),
            _prediction(0.8, 0),
            _prediction(0.3, 1),  # below breakeven, excluded
        ]
        result = simulate_bets(preds, decimal_odds=2.0)
        # Bets placed on 0.6 and 0.8
        assert result.predicted_prob_mean == pytest.approx(0.7, abs=0.001)
        assert result.total_bets == 2

    def test_daily_profits_recorded(self):
        preds = [
            _prediction(0.9, 1, d="2025-04-01"),
            _prediction(0.9, 0, d="2025-04-02"),
            _prediction(0.9, 1, d="2025-04-01"),  # same day as first
        ]
        result = simulate_bets(preds, decimal_odds=2.0, stake_per_bet=1.0)
        # 2 bets on Apr 1: +$1 and +$1 = cumulative $2
        # 1 bet on Apr 2: -$1 = cumulative $1
        assert len(result.daily_profits) == 2
        assert result.daily_profits[0] == pytest.approx(2.0)
        assert result.daily_profits[1] == pytest.approx(1.0)

    def test_win_rate_accuracy(self):
        preds = [
            _prediction(0.9, 1),
            _prediction(0.9, 0),
            _prediction(0.9, 1),
            _prediction(0.9, 1),
        ]
        result = simulate_bets(preds, decimal_odds=2.0)
        assert result.win_rate == 0.75


# ---------------------------------------------------------------------------
# calibration_buckets
# ---------------------------------------------------------------------------


class TestCalibrationBuckets:
    def test_perfect_calibration(self):
        preds = [
            _prediction(0.2, 0),
            _prediction(0.2, 0),
            _prediction(0.2, 1),  # 33% observed in 0.2 bin
            _prediction(0.8, 1),
            _prediction(0.8, 1),
            _prediction(0.8, 0),  # 67% observed in 0.8 bin
        ]
        buckets = calibration_buckets(preds, n_bins=10)
        assert len(buckets) >= 2

        for b in buckets:
            if b["bin_lower"] <= 0.2 < b["bin_upper"]:
                assert b["observed_freq"] == pytest.approx(1 / 3, abs=0.01)
            if b["bin_lower"] <= 0.8 < b["bin_upper"]:
                assert b["observed_freq"] == pytest.approx(2 / 3, abs=0.01)

    def test_empty_list(self):
        assert calibration_buckets([]) == []

    def test_all_same_bin(self):
        preds = [_prediction(0.5, 1), _prediction(0.5, 0), _prediction(0.5, 1)]
        buckets = calibration_buckets(preds, n_bins=10)
        mid_bin = [b for b in buckets if b["bin_lower"] <= 0.5 < b["bin_upper"]]
        assert len(mid_bin) == 1
        assert mid_bin[0]["count"] == 3
        assert mid_bin[0]["observed_freq"] == pytest.approx(2 / 3, abs=0.01)

    def test_excludes_empty_bins(self):
        preds = [_prediction(0.95, 1), _prediction(0.95, 0)]
        buckets = calibration_buckets(preds, n_bins=10)
        # Only the 0.9-1.0 bin should be populated
        for b in buckets:
            if b["bin_lower"] < 0.9:
                assert False, f"Empty bin [{b['bin_lower']},{b['bin_upper']}) included"


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------


class TestExpectedCalibrationError:
    def test_perfect_calibration(self):
        # 0.2 bin: 1 of 5 positive = 20% observed (matches mean_pred=0.2)
        preds_20 = [_prediction(0.2, 1)] + [_prediction(0.2, 0) for _ in range(4)]
        # 0.8 bin: 4 of 5 positive = 80% observed (matches mean_pred=0.8)
        preds_80 = [_prediction(0.8, 1) for _ in range(4)] + [_prediction(0.8, 0)]
        ece = expected_calibration_error(preds_20 + preds_80, n_bins=10)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_imperfect_calibration(self):
        # 0.2 bin: predicted 0.2 but observed 0.0 (error = 0.2)
        # 0.8 bin: predicted 0.8 but observed 1.0 (error = 0.2)
        preds = [
            _prediction(0.2, 0),
            _prediction(0.2, 0),
            _prediction(0.2, 0),
            _prediction(0.8, 1),
            _prediction(0.8, 1),
            _prediction(0.8, 1),
        ]
        # Each bin has count=3, total=6
        # Error contribution: 3*0.2 + 3*0.2 = 1.2, /6 = 0.2
        ece = expected_calibration_error(preds, n_bins=10)
        assert ece == pytest.approx(0.2, abs=0.01)

    def test_empty_list(self):
        assert expected_calibration_error([]) == 0.0

    def test_single_bin(self):
        preds = [_prediction(0.5, 1), _prediction(0.5, 0)]
        ece = expected_calibration_error(preds, n_bins=10)
        # mean_predicted = 0.5, observed_freq = 0.5, error = 0.0
        assert ece == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_strictly_increasing(self):
        assert max_drawdown([1, 2, 3, 4, 5]) == 0.0

    def test_single_element(self):
        assert max_drawdown([100.0]) == 0.0

    def test_empty(self):
        assert max_drawdown([]) == 0.0

    def test_simple_decline(self):
        mdd = max_drawdown([100, 90, 80])
        assert mdd == pytest.approx(0.2, abs=0.001)  # 100 -> 80 = 20%

    def test_recovery_after_drawdown(self):
        mdd = max_drawdown([100, 90, 80, 95])
        # peak 100 -> trough 80 = 20%
        assert mdd == pytest.approx(0.2, abs=0.001)

    def test_multiple_peaks(self):
        mdd = max_drawdown([100, 200, 250, 150, 300])
        # peak 250 -> trough 150 = 40%
        assert mdd == pytest.approx(0.4, abs=0.001)

    def test_two_elements_decline(self):
        mdd = max_drawdown([100, 80])
        assert mdd == pytest.approx(0.2, abs=0.001)

    def test_two_elements_increase(self):
        assert max_drawdown([80, 100]) == 0.0

    def test_flat(self):
        assert max_drawdown([50, 50, 50]) == 0.0


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


class TestBetResultDefaults:
    def test_empty_result(self):
        r = BetResult()
        assert r.total_bets == 0
        assert r.total_profit == 0.0
        assert r.roi == 0.0
        assert r.max_drawdown == 0.0

    def test_fields_default_correctly(self):
        r = BetResult(total_bets=5, wins=3, losses=2, total_profit=10.0)
        assert r.losses == 2
        assert r.win_rate == 0.0  # not auto-calculated from fields


class TestGamePrediction:
    def test_basic_creation(self):
        gp = GamePrediction(
            date=date(2025, 4, 1),
            player_id=42,
            game_pk=1000,
            predicted_prob=0.75,
            actual=1,
            hits=2,
            target_col="target_1.5",
        )
        assert gp.date == date(2025, 4, 1)
        assert gp.player_id == 42
        assert gp.game_pk == 1000
        assert gp.predicted_prob == 0.75
        assert gp.actual == 1
        assert gp.hits == 2
        assert gp.target_col == "target_1.5"


# ---------------------------------------------------------------------------
# Calibrator save/load
# ---------------------------------------------------------------------------


class TestCalibratorPersistence:
    def test_save_load_cycle(self, tmp_path):
        preds = [
            GamePrediction(date=date(2021, 4, 1), player_id=1, game_pk=100,
                           predicted_prob=0.3, actual=0, hits=0, target_col="target_0.5"),
            GamePrediction(date=date(2021, 4, 1), player_id=1, game_pk=101,
                           predicted_prob=0.7, actual=1, hits=1, target_col="target_0.5"),
            GamePrediction(date=date(2022, 4, 1), player_id=1, game_pk=102,
                           predicted_prob=0.4, actual=0, hits=0, target_col="target_0.5"),
            GamePrediction(date=date(2022, 4, 1), player_id=1, game_pk=103,
                           predicted_prob=0.8, actual=1, hits=1, target_col="target_0.5"),
        ]
        calibrators = fit_season_calibrators(preds)
        assert set(calibrators) == {2021, 2022}
        save_calibrators(calibrators, str(tmp_path))
        loaded = load_calibrators(str(tmp_path))
        assert set(loaded) == {2021, 2022}
        cal_preds = apply_calibrators(preds, loaded)
        assert len(cal_preds) == 4
        for cp in cal_preds:
            assert 0.0 <= cp.predicted_prob <= 1.0
