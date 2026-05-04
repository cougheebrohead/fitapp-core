"""BMR, TDEE, macro target math.

Mifflin-St Jeor (1990) is the most accurate non-clinical BMR predictor
for adults; agrees with Katch-McArdle to within a few % for users who
don't have body-fat % handy. Activity factors and goal modifiers from
Academy of Nutrition and Dietetics 2024 guidelines.
"""

from __future__ import annotations

from typing import Literal

Sex = Literal["male", "female"]
Activity = Literal["sedentary", "light", "moderate", "active", "very_active"]
Goal = Literal["lose", "maintain", "gain"]

ACTIVITY_FACTOR: dict[str, float] = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}

# Calorie delta from maintenance, in kcal/day. Conservative deltas — aim
# for ~0.5 kg/week change which is the upper-tolerable limit for body
# composition without muscle loss / metabolic adaptation.
GOAL_DELTA: dict[str, int] = {"lose": -500, "maintain": 0, "gain": 300}

# Lower bounds — calorie targets below these correlate with disordered
# eating risk in the literature. Cap deficits even when user requests more.
MIN_CALORIES_FEMALE = 1200
MIN_CALORIES_MALE = 1500


def bmr_mifflin(
    age: int,
    sex: Sex,
    weight_kg: float,
    height_cm: float,
) -> float:
    """Mifflin-St Jeor basal metabolic rate (kcal/day).

    Male:   10*W + 6.25*H - 5*A + 5
    Female: 10*W + 6.25*H - 5*A - 161
    """
    if age < 1 or age > 120:
        raise ValueError(f"age out of range: {age}")
    if weight_kg <= 0 or weight_kg > 500:
        raise ValueError(f"weight_kg out of range: {weight_kg}")
    if height_cm <= 0 or height_cm > 280:
        raise ValueError(f"height_cm out of range: {height_cm}")
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return base + 5 if sex == "male" else base - 161


def tdee(bmr: float, activity: Activity = "moderate") -> float:
    """Total daily energy expenditure (kcal/day) = BMR × activity factor."""
    factor = ACTIVITY_FACTOR.get(activity)
    if factor is None:
        raise ValueError(f"unknown activity level: {activity}")
    return bmr * factor


def macro_targets(
    tdee_value: float,
    goal: Goal,
    weight_kg: float,
    sex: Sex = "female",
) -> dict[str, int | float]:
    """Daily macro targets given TDEE + goal.

    Protein anchored to 1.6 g/kg lean body weight (we use total weight as
    a conservative proxy without body-fat data — this is the AND-CSS-IOC
    consensus floor for active adults). Fat is 25% of total kcal (lower
    bound for hormone health). Carbs fill the remainder.
    """
    target_kcal = max(0, int(tdee_value + GOAL_DELTA.get(goal, 0)))
    floor = MIN_CALORIES_FEMALE if sex == "female" else MIN_CALORIES_MALE
    target_kcal = max(target_kcal, floor)

    protein_g = round(weight_kg * 1.6)
    fat_kcal = target_kcal * 0.25
    fat_g = round(fat_kcal / 9)

    carb_kcal = max(0, target_kcal - (protein_g * 4) - fat_kcal)
    carb_g = round(carb_kcal / 4)

    return {
        "calories": target_kcal,
        "protein": protein_g,
        "carbs": carb_g,
        "fat": fat_g,
        "water_oz": water_target(weight_kg),
    }


def water_target(weight_kg: float) -> int:
    """Daily water target in fluid ounces. ~50% of body weight in lbs is the
    common rule of thumb; we floor at 64oz (the 8-cup baseline) and cap at
    150oz to discourage hyponatremia in very large athletes."""
    weight_lbs = weight_kg * 2.20462
    oz = round(weight_lbs * 0.5)
    return max(64, min(150, oz))
