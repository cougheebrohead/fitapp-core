from fitapp_core import allergen_alerts, condition_flags


def test_anaphylactic_allergy_triggers_stop():
    items = [{"name": "Peanut butter sandwich", "ingredients_text": "peanut butter, bread"}]
    allergies = [{"allergen": "peanut", "severity": "anaphylactic", "notes": ""}]
    alerts = allergen_alerts(items, allergies)
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "stop"
    assert "peanut" in alerts[0]["title"].lower()


def test_hidden_milk_via_casein():
    items = [{"name": "Protein bar", "ingredients_text": "casein protein, oats"}]
    allergies = [{"allergen": "milk", "severity": "moderate", "notes": ""}]
    alerts = allergen_alerts(items, allergies)
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "medium"


def test_no_allergy_no_alert():
    items = [{"name": "Grilled chicken", "ingredients_text": "chicken, salt"}]
    allergies = [{"allergen": "peanut", "severity": "anaphylactic", "notes": ""}]
    assert allergen_alerts(items, allergies) == []


def test_t2d_high_sugar_flag():
    flags = condition_flags({"sugar": 50, "carbs": 70, "calories": 600}, ["t2d"])
    assert len(flags) == 1
    assert flags[0]["code"] == "condition_t2d"


def test_hypertension_high_sodium():
    flags = condition_flags({"sodium": 2000, "calories": 800}, ["hypertension"])
    assert len(flags) == 1


def test_no_condition_no_flag():
    flags = condition_flags({"sugar": 50}, [])
    assert flags == []
