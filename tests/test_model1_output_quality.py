from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_kavach_model1 import analyze_answer
from bpe import BPE
from bpe_tokenizer import BPETokenizer
from kavach_model1 import utf8_valid_token_mask
import torch


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



def test_bpe_token_bytes_round_trip_preserves_devanagari_utf8():
    tok = BPETokenizer(BPE([(0xE0, 0xA4), (256, 0xB9)]))
    text = "ह"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert b"".join(tok.token_bytes(i) for i in ids) == text.encode("utf-8")


def test_utf8_mask_allows_incomplete_prefix_but_not_invalid_byte():
    tok = BPETokenizer(BPE([]))
    mask = utf8_valid_token_mask(tok, [], torch.device("cpu"))
    assert bool(mask[0xE0])  # lead byte may be completed by a later token
    assert bool(mask[ord("A")])
    mask_after_lead = utf8_valid_token_mask(tok, [0xE0], torch.device("cpu"))
    assert not bool(mask_after_lead[ord("A")])  # ASCII cannot follow a 3-byte lead
    assert bool(mask_after_lead[0xA4])
    assert not bool(mask_after_lead[tok.eos])


def test_utf8_mask_only_allows_eos_at_character_boundary():
    tok = BPETokenizer(BPE([]))
    complete = utf8_valid_token_mask(tok, [ord("A")], torch.device("cpu"))
    incomplete = utf8_valid_token_mask(tok, [0xE0], torch.device("cpu"))
    assert bool(complete[tok.eos])
    assert not bool(incomplete[tok.eos])
