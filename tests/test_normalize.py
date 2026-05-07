from __future__ import annotations

import pytest

from radiotalk.normalize import normalize
from radiotalk.normalize.airports import expand_airport_codes
from radiotalk.normalize.callsigns import expand_callsigns
from radiotalk.normalize.numbers import (
    expand_altimeters,
    expand_decimals,
    expand_flight_levels,
    expand_frequencies,
    expand_headings,
    expand_remaining_numbers,
    expand_squawks,
    expand_winds,
    normalize_numbers,
)
from radiotalk.normalize.phonetic import spell_digits, spell_letters
from radiotalk.normalize.runways import expand_runways


class TestPhonetic:
    def test_digits(self):
        assert spell_digits("9462") == "nine four six two"
        assert spell_digits("0") == "zero"

    def test_letters(self):
        assert spell_letters("AB") == "alpha bravo"
        assert spell_letters("XY") == "x-ray yankee"


class TestCallsigns:
    def test_known_airline(self):
        assert expand_callsigns("DLH9462") == "lufthansa nine four six two"

    def test_lufthansa_in_sentence(self):
        out = expand_callsigns("DLH9462, ready for push.")
        assert out == "lufthansa nine four six two, ready for push."

    def test_n_number(self):
        assert expand_callsigns("N79872") == "november seven nine eight seven two"

    def test_military_already_word(self):
        assert expand_callsigns("ARMY7361") == "army seven three six one"

    def test_unknown_prefix_falls_back_to_nato(self):
        # ZZZ isn't in the lookup; should NATO-spell.
        assert expand_callsigns("ZZZ123") == "zulu zulu zulu one two three"

    def test_already_spelled_is_noop(self):
        s = "lufthansa nine four six two heavy"
        assert expand_callsigns(s) == s

    def test_space_separated_known_airline(self):
        assert expand_callsigns("ACA 7490") == "air canada seven four nine zero"

    def test_space_separated_unknown_left_alone(self):
        # 'IAS' is indicated airspeed, not an airline. Don't mangle it.
        assert expand_callsigns("IAS 220") == "IAS 220"

    def test_aircraft_type_with_space_left_alone(self):
        # 'Boeing 737' / 'Cessna 172' should never be touched by the
        # space-separated rule (mixed-case prefix is excluded).
        assert expand_callsigns("Boeing 737") == "Boeing 737"
        assert expand_callsigns("Cessna 172") == "Cessna 172"

    def test_word_digits_known_airline(self):
        # LLM emits this mixed style for a non-trivial slice of the corpus.
        assert expand_callsigns("AFR four two three three") == "air france four two three three"
        assert expand_callsigns("QFA four three seven three") == "qantas four three seven three"

    def test_word_digits_unknown_prefix_left_alone(self):
        # Random caps tokens like 'ATC', 'IFR', 'DME' aren't airlines.
        assert expand_callsigns("ATC four two") == "ATC four two"
        assert expand_callsigns("DME four miles") == "DME four miles"

    def test_word_digits_inside_sentence(self):
        s = "AFR four two three three, ready for push"
        assert expand_callsigns(s) == "air france four two three three, ready for push"


class TestAirports:
    def test_known_us_airport(self):
        assert expand_airport_codes("5 miles west of KSFO") == "5 miles west of San Francisco"

    def test_kennedy(self):
        assert expand_airport_codes("cleared into KJFK") == "cleared into New York"

    def test_unknown_4letter_left_alone(self):
        # ARMY isn't an ICAO airport — leave for other rules / TTS.
        assert expand_airport_codes("ARMY7361 heavy") == "ARMY7361 heavy"

    def test_three_letter_not_matched(self):
        # 3-letter sequences (airline prefixes, callsign suffixes) ignored.
        assert expand_airport_codes("DLH heavy") == "DLH heavy"

    def test_iata_facility(self):
        assert expand_airport_codes("contact SFO Tower") == "contact San Francisco Tower"
        assert expand_airport_codes("DFW Center") == "Dallas-Fort Worth Center"
        assert expand_airport_codes("MEM Approach") == "Memphis Approach"

    def test_iata_three_letter_without_facility(self):
        # IFR, ATC, GPS, ILS — bare 3-letter caps without facility word stays.
        assert expand_airport_codes("IFR cleared") == "IFR cleared"


class TestRunways:
    def test_with_side(self):
        assert expand_runways("Runway 34R") == "runway three four right"

    def test_without_side(self):
        assert expand_runways("Runway 25") == "runway two five"

    def test_with_space_before_side(self):
        assert expand_runways("runway 23 C") == "runway two three center"

    def test_left(self):
        assert expand_runways("Runway 09L") == "runway zero nine left"

    def test_already_spelled_is_noop(self):
        s = "runway two three charlie"
        assert expand_runways(s) == s

    def test_bare_designator_takeoff_from(self):
        assert expand_runways("takeoff from 35R") == "takeoff from three five right"

    def test_bare_designator_hold_short(self):
        assert expand_runways("hold short 16R") == "hold short one six left" or \
            expand_runways("hold short 16R") == "hold short one six right"
        # The above is intentionally lax — verifying R->right next.
        assert expand_runways("hold short 16R") == "hold short one six right"

    def test_bare_designator_at_start(self):
        assert expand_runways("08L line up and wait") == "zero eight left line up and wait"

    def test_bare_designator_for_ils(self):
        assert expand_runways("for ILS 26C, ten miles") == "for ILS two six center, ten miles"


class TestFrequencies:
    def test_basic(self):
        assert expand_frequencies("contact tower 118.7") == "contact tower one one eight point seven"

    def test_three_decimals(self):
        assert expand_frequencies("125.750") == "one two five point seven five zero"

    def test_idempotent(self):
        s = "one two five point seven"
        assert expand_frequencies(s) == s


class TestSquawks:
    def test_squawk(self):
        assert expand_squawks("squawk 3560") == "squawk three five six zero"

    def test_squawking(self):
        assert expand_squawks("squawking 4402") == "squawking four four zero two"


class TestHeadings:
    def test_heading_three_digits(self):
        assert expand_headings("heading 115") == "heading one one five"

    def test_heading_pads_two_digits(self):
        assert expand_headings("heading 90") == "heading zero nine zero"


class TestWinds:
    def test_wind(self):
        assert expand_winds("wind 170 at 10") == "wind one seven zero at one zero"

    def test_winds_plural(self):
        assert expand_winds("winds 020 at 5") == "winds zero two zero at zero five"


class TestAltimeters:
    def test_altimeter(self):
        assert expand_altimeters("altimeter 30.02") == "altimeter three zero zero two"


class TestFlightLevels:
    def test_with_space(self):
        assert expand_flight_levels("FL 250") == "flight level two five zero"

    def test_without_space(self):
        assert expand_flight_levels("FL250") == "flight level two five zero"


class TestDecimals:
    def test_visibility(self):
        assert expand_decimals("visibility 1.6 miles") == "visibility one point six miles"


class TestRemainingNumbers:
    def test_altitude_thousands(self):
        assert expand_remaining_numbers("4000") == "four thousand"

    def test_altitude_thousand_five_hundred(self):
        assert expand_remaining_numbers("4500") == "four thousand five hundred"

    def test_altitude_two_digit_thousands(self):
        assert expand_remaining_numbers("12000") == "one two thousand"
        assert expand_remaining_numbers("12500") == "one two thousand five hundred"

    def test_three_digit_bare(self):
        assert expand_remaining_numbers("of 115") == "of one one five"

    def test_round_hundreds(self):
        assert expand_remaining_numbers("100 feet") == "one hundred feet"
        assert expand_remaining_numbers("200 feet") == "two hundred feet"
        assert expand_remaining_numbers("500") == "five hundred"
        assert expand_remaining_numbers("900") == "nine hundred"

    def test_three_digit_non_round_stays_digit_by_digit(self):
        # Headings, frequencies, etc. are mid-100s — keep digit form.
        assert expand_remaining_numbers("250") == "two five zero"
        assert expand_remaining_numbers("115") == "one one five"

    def test_small_grammatical(self):
        assert expand_remaining_numbers("5 miles") == "five miles"

    def test_two_digit(self):
        assert expand_remaining_numbers("at 10 o'clock") == "at ten o'clock"


class TestNumbersPipeline:
    def test_strips_thousands_separator(self):
        assert normalize_numbers("climb to 10,000") == "climb to one zero thousand"

    def test_squawk_then_altitude(self):
        s = "squawk 3560, climb 4000"
        assert normalize_numbers(s) == "squawk three five six zero, climb four thousand"

    def test_bare_4digit_code_digit_by_digit(self):
        # Squawk-shaped 4-digit numbers without the 'squawk' word should be
        # digit-by-digit, not altitude form.
        assert normalize_numbers("ident 5562") == "ident five five six two"
        assert normalize_numbers("ceiling 1176") == "ceiling one one seven six"

    def test_4digit_round_thousand_stays_altitude_form(self):
        # 1500, 4000, 10000 — real altitudes — keep altitude phraseology.
        assert normalize_numbers("4500") == "four thousand five hundred"
        assert normalize_numbers("1500") == "one thousand five hundred"
        assert normalize_numbers("10500") == "one zero thousand five hundred"


class TestAltimeterBare:
    def test_altimeter_no_decimal(self):
        assert normalize_numbers("altimeter 3008") == "altimeter three zero zero eight"

    def test_qnh(self):
        assert normalize_numbers("QNH 1013") == "qnh one zero one three"


class TestLowercaseCaps:
    def test_lowercases_all_caps_words(self):
        # 4+ letter caps words get lowercased.
        assert normalize("WIND TWO SEVEN ZERO AT FIVE KNOTS") == \
            "wind two seven zero at five knots"

    def test_lowercases_runway_text(self):
        assert normalize("RUNWAY 09 IN USE") == "runway zero nine in use"

    def test_lowercases_2_letter_words(self):
        # 2-letter caps words also lowercase (AT, IN, OK, etc.)
        out = normalize("HOLD AT TAXIWAY ALPHA")
        assert "at" in out and "AT" not in out

    def test_callsigns_still_expand_before_lowercase(self):
        # Callsign expansion runs before the lowercase rule, so the
        # uppercase prefix matches the airline lookup.
        assert normalize("DLH9462, RUNWAY 09") == \
            "lufthansa nine four six two, runway zero nine"

    def test_airport_still_expands(self):
        # Proper-noun city names from airportsdata stay mixed-case (not all-caps,
        # so the lowercase rule leaves them alone).
        assert normalize("CONTACT KSFO TOWER") == "contact San Francisco tower"


class TestEndToEnd:
    @pytest.mark.parametrize("raw, expected_substrings", [
        (
            "DLH9462, Denver ground, ready for push, squawking 4402, heavy.",
            ["lufthansa nine four six two", "squawking four four zero two", "heavy"],
        ),
        (
            "Cleared to land Runway 35, contact tower 118.7, altimeter 30.02.",
            ["runway three five", "one one eight point seven", "altimeter three zero zero two"],
        ),
        (
            "N885, climb and maintain 4000, squawk 3560.",
            ["november eight eight five", "four thousand", "squawk three five six zero"],
        ),
        (
            "Wind 170 at 10, visibility 1.6 miles, altimeter 30.11.",
            ["Wind one seven zero at one zero", "one point six miles", "altimeter three zero one one"],
        ),
        (
            "Climb FL250, heading 115, traffic at 10 o'clock, 5 miles.",
            ["flight level two five zero", "heading one one five", "ten o'clock", "five miles"],
        ),
    ])
    def test_real_corpus_examples(self, raw, expected_substrings):
        out = normalize(raw)
        for sub in expected_substrings:
            assert sub in out, f"expected {sub!r} in {out!r}"

    def test_idempotent(self):
        raw = "DLH9462 climb 4000 squawk 3560 contact tower 118.7"
        once = normalize(raw)
        twice = normalize(once)
        assert once == twice, f"\n once: {once!r}\n twice: {twice!r}"
