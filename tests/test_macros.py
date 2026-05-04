import pytest

from fitapp_core import bmr_mifflin, tdee, macro_targets, water_target


# Reference values from Mifflin-St Jeor (1990) examples
def test_bmr_male_30yo():
    # 30 yo male, 80 kg, 175 cm → 1755 kcal
    assert round(bmr_mifflin(30, "male", 80, 175)) == 1755


def test_bmr_female_30yo():
    # 30 yo female, 65 kg, 165 cm → 1389 kcal
    assert round(bmr_mifflin(30, "female", 65, 165)) == 1389


def test_tdee_moderate():
    bmr = bmr_mifflin(30, "male", 80, 175)
    assert round(tdee(bmr, "moderate")) == 2720


def test_macro_targets_balance():
    out = macro_targets(2400, "maintain", 80, "male")
    # 4*P + 4*C + 9*F should be within ~2% of target
    sum_kcal = out["protein"] * 4 + out["carbs"] * 4 + out["fat"] * 9
    assert abs(sum_kcal - 2400) / 2400 < 0.05


def test_lose_floor_female():
    """Aggressive female user with low TDEE should not drop below floor."""
    out = macro_targets(1500, "lose", 50, "female")  # would be 1000 without floor
    assert out["calories"] >= 1200


def test_water_target_bounds():
    assert water_target(50) >= 64    # small person floor
    assert water_target(150) <= 150  # very large athlete cap


def test_invalid_inputs():
    with pytest.raises(ValueError):
        bmr_mifflin(0, "male", 80, 175)
    with pytest.raises(ValueError):
        bmr_mifflin(30, "male", 0, 175)
    with pytest.raises(ValueError):
        tdee(1500, "invalid_activity")  # type: ignore[arg-type]
