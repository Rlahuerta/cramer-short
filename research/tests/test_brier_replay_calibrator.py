from research.models.brier_replay_calibrator import BrierReplayCalibrator


def _brier_score(rows: list[dict[str, float | int]]) -> float:
    if not rows:
        return 0.0
    return sum((row["probability"] - row["actual"]) ** 2 for row in rows) / len(rows)


def _run_replay(
    calibrator: BrierReplayCalibrator,
    rows: list[dict[str, float | int]],
    burn_in_fraction: float = 0.5,
) -> dict[str, float]:
    raw_holdout: list[dict[str, float | int]] = []
    calibrated_holdout: list[dict[str, float | int]] = []
    raw_mid: list[dict[str, float | int]] = []
    calibrated_mid: list[dict[str, float | int]] = []
    holdout_start = int(len(rows) * burn_in_fraction)

    for index, row in enumerate(rows):
        raw = float(row["raw"])
        actual = int(row["actual"])
        calibrated = calibrator.predict(raw)
        if index >= holdout_start:
            raw_holdout.append({"probability": raw, "actual": actual})
            calibrated_holdout.append({"probability": calibrated, "actual": actual})
            if 0.4 <= raw <= 0.6:
                raw_mid.append({"probability": raw, "actual": actual})
                calibrated_mid.append({"probability": calibrated, "actual": actual})
        calibrator.record(raw, actual)

    return {
        "raw_holdout_brier": _brier_score(raw_holdout),
        "calibrated_holdout_brier": _brier_score(calibrated_holdout),
        "raw_mid_brier": _brier_score(raw_mid),
        "calibrated_mid_brier": _brier_score(calibrated_mid),
    }


def test_brier_replay_calibrator_improves_holdout_brier() -> None:
    replay = []
    for index in range(120):
        raw = 0.58 if index % 4 < 2 else 0.42
        actual = 1 if index % 4 < 2 and index % 6 == 0 else 0
        if index % 4 >= 2:
            actual = 0 if index % 6 == 0 else 1
        replay.append({"raw": raw, "actual": actual})

    calibrator = BrierReplayCalibrator(
        learning_rate=0.1,
        mid_confidence_weight=4,
        max_slope=3,
    )
    result = _run_replay(calibrator, replay)

    assert result["calibrated_holdout_brier"] < result["raw_holdout_brier"]
    assert result["calibrated_mid_brier"] < result["raw_mid_brier"]
    assert calibrator.state()["slope"] < 1.0


def test_brier_replay_calibrator_stays_close_to_identity() -> None:
    replay = []
    for index in range(120):
        raw = 0.7 if index % 2 == 0 else 0.3
        actual = 1 if index % 10 < 7 else 0
        replay.append({"raw": raw, "actual": actual if index % 2 == 0 else 1 - actual})

    calibrator = BrierReplayCalibrator(
        learning_rate=0.05,
        mid_confidence_weight=2,
        max_slope=2,
    )
    result = _run_replay(calibrator, replay)
    state = calibrator.state()

    assert abs(state["bias"]) < 0.2
    assert abs(state["slope"] - 1.0) < 0.2
    assert result["calibrated_holdout_brier"] <= result["raw_holdout_brier"] + 0.01
