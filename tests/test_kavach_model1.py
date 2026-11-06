from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kavach_model1 import KavachModel1
from capability_policy import CapabilityPolicy


def test_kavach_model1_profile_is_least_privilege():
    policy = CapabilityPolicy("cybersecurity_assistant")
    assert policy.policy.granted == {"answer_question", "read_document"}
    assert "execute_code" in policy.policy.denied
    assert "network_out" in policy.policy.denied


def test_chat_prompt_uses_language_tag_and_special_tokens():
    class FakeTokenizer:
        stoi = {"<bos>": 1, "<user>": 2, "<sep>": 3, "<assistant>": 4,
                "<hi>": 5, "<en>": 6, "<mix>": 7}
        def encode(self, text):
            return [10, 11, 12]

    adapter = object.__new__(KavachModel1)
    adapter.tokenizer = FakeTokenizer()
    hindi = adapter._prompt_ids("साइबर सुरक्षा क्या है?")
    english = adapter._prompt_ids("What is network security?")
    assert hindi[:2] == [1, 2] and hindi[-2:] == [4, 5]
    assert english[:2] == [1, 2] and english[-2:] == [4, 6]


def test_hindi_and_hinglish_questions_are_not_code_payloads():
    from kavach import Kavach
    assert Kavach._is_question("SQL injection क्या है?")
    assert Kavach._is_question("kya firewall kaam karta hai?")
    assert not Kavach._is_question("import os; os.system('id')")
