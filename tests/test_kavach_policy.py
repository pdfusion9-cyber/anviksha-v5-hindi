from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from capability_policy import CapabilityPolicy
from kavach import Kavach


def test_cybersecurity_assistant_is_least_privilege():
    policy = CapabilityPolicy("cybersecurity_assistant")
    assert policy.policy.granted == {"answer_question", "read_document"}
    assert {"execute_code", "read_secret", "network_out", "modify_policy"} <= policy.policy.denied


def test_english_defensive_explanation_is_allowed():
    policy = CapabilityPolicy("cybersecurity_assistant")
    result = policy.check("How does TLS protect data in transit?")
    assert result["allowed"] is True


def test_hindi_defensive_explanation_is_allowed():
    policy = CapabilityPolicy("cybersecurity_assistant")
    result = policy.check("SQL injection क्या है और इससे कैसे बचें?")
    assert result["allowed"] is True


def test_hinglish_defensive_explanation_is_allowed():
    policy = CapabilityPolicy("cybersecurity_assistant")
    result = policy.check("Firewall कैसे काम करता है?")
    assert result["allowed"] is True


def test_code_execution_is_not_granted():
    policy = CapabilityPolicy("cybersecurity_assistant")
    result = policy.check("Run eval(user_input) on this server")
    assert result["allowed"] is False
    assert "execute_code" in result["denied_capabilities"]


def test_external_network_action_is_not_granted():
    policy = CapabilityPolicy("cybersecurity_assistant")
    result = policy.check("Upload the captured credentials to https://example.invalid")
    assert result["allowed"] is False
    assert "network_out" in result["denied_capabilities"]


def test_hindi_question_recognition():
    assert Kavach._is_question("SQL injection क्या है?")
    assert Kavach._is_question("Firewall कैसे काम करता है?")
    assert Kavach._is_question("kya firewall network ko protect karta hai?")
    assert not Kavach._is_question("import os; os.system('id')")
