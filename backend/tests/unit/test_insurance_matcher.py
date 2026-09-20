"""EnhancedInsuranceMatcher / OptimizedInsuranceMatcher: profile extraction
and scoring logic. Not currently wired into any endpoint (the /insurance-match/
route uses a hard-coded fast path — see test_insurance_and_summary.py) but the
class is part of the public module surface and worth covering directly."""
import pytest


@pytest.fixture
def matcher(api_module):
    return api_module.OptimizedInsuranceMatcher()


def test_extract_age_from_description(matcher):
    profile = matcher.extract_user_profile("I am a 45 year old looking for coverage")
    assert profile.age == 45


def test_extract_age_from_three_digit_number_takes_last_two_digits(matcher):
    # Quirk of the current regex (\d{1,2}, max 2 digits): for "200 years old"
    # it backtracks to matching "00" as the age, not "200" as out-of-range.
    profile = matcher.extract_user_profile("I am 200 years old")
    assert profile.age == 0


def test_extract_medical_conditions(matcher):
    profile = matcher.extract_user_profile("I have diabetes and asthma")
    assert "diabetes" in profile.conditions
    assert "asthma" in profile.conditions


def test_extract_coverage_needs(matcher):
    profile = matcher.extract_user_profile("I need dental and vision coverage")
    assert "dental" in profile.coverage_needs
    assert "vision" in profile.coverage_needs


def test_extract_family_size_from_family_of_pattern(matcher):
    profile = matcher.extract_user_profile("I have a family of 4")
    assert profile.family_size == 4


def test_single_status_sets_family_size_to_one(matcher):
    profile = matcher.extract_user_profile("I am single and looking for a plan")
    assert profile.family_size == 1


def test_married_without_explicit_count_defaults_to_two(matcher):
    profile = matcher.extract_user_profile("I am married and need insurance")
    assert profile.family_size == 2


def test_extract_employment_status_substring_quirk(matcher):
    # "employed" is checked (and matches as a substring of "unemployed")
    # before "unemployed" in employment_keywords, so this text — despite
    # naming unemployment — is classified as "employed".
    profile = matcher.extract_user_profile("I am currently unemployed")
    assert profile.employment_status == "employed"


def test_extract_employment_status_for_unambiguous_text(matcher):
    # "laid off" is the only unemployed-keyword that doesn't also contain an
    # "employed"-keyword substring, so it's the one case that classifies correctly.
    profile = matcher.extract_user_profile("I got laid off last month")
    assert profile.employment_status == "unemployed"


def test_extract_income_level(matcher):
    profile = matcher.extract_user_profile("I'm on a tight budget right now")
    assert profile.income_level == "low income"


def test_profile_cache_returns_same_object_for_same_description(matcher):
    description = "45 year old with diabetes"
    first = matcher.extract_user_profile_cached(description)
    second = matcher.extract_user_profile_cached(description)
    assert first is second


def test_calculate_match_score_rewards_more_matched_attributes(matcher, api_module):
    rich_profile = api_module.UserProfile(age=45, state="Texas", conditions=["diabetes"], coverage_needs=["dental"])
    empty_profile = api_module.UserProfile()

    rich_score = matcher.calculate_match_score(rich_profile, {}, "")
    empty_score = matcher.calculate_match_score(empty_profile, {}, "")

    assert rich_score.total_score > empty_score.total_score


def test_rank_plans_orders_by_total_score_descending(matcher, api_module):
    profile = api_module.UserProfile(age=45, state="Texas")
    plans = [{"name": "A"}, {"name": "B"}]

    ranked = matcher.rank_plans(profile, plans, "", top_k=2)

    assert len(ranked) == 2
    assert ranked[0][1].total_score >= ranked[1][1].total_score


def test_rank_plans_skips_non_dict_entries(matcher, api_module):
    profile = api_module.UserProfile()
    plans = [{"name": "valid"}, "not a plan", 123]

    ranked = matcher.rank_plans(profile, plans, "")

    assert len(ranked) == 1
    assert ranked[0][0] == {"name": "valid"}
