from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_kavach_model1 import analyze_answer


def test_repeated_english_token_is_not_usable():
    result = analyze_answer("en_firewall", "the the the the the the the the")
    assert result["passed"] is False
    assert result["checks"]["not_dominated_by_one_word"] is False


def test_replacement_characters_fail_hindi_output():
    result = analyze_answer("hi_firewall", "������������������������")
    assert result["passed"] is False
    assert result["checks"]["valid_unicode_no_replacement_chars"] is False


def test_english_answer_requires_latin_script():
    result = analyze_answer("en_firewall", "सुरक्षा नेटवर्क फ़ायरवॉल सुरक्षा नेटवर्क फ़ायरवॉल")
    assert result["passed"] is False
    assert result["checks"]["expected_latin_script"] is False


def test_hindi_answer_requires_devanagari_script():
    result = analyze_answer("hi_firewall", "Firewall protects network traffic from unauthorized access and threats.")
    assert result["passed"] is False
    assert result["checks"]["expected_devanagari_script"] is False
