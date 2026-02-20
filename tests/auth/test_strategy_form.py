from backend.auth0login.forms import StrategyForm


def test_strategy_form_cleaned_data_is_dict():
    form = StrategyForm(data={"strategy": "manual"})

    assert form.is_valid()
    assert isinstance(form.cleaned_data, dict)
    assert form.cleaned_data["strategy"] == "manual"


def test_strategy_form_rejects_unknown_choice():
    form = StrategyForm(data={"strategy": "not_a_real_strategy"})

    assert not form.is_valid()
    assert "strategy" in form.errors
