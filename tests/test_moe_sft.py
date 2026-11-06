from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_sft import build_example, make_batch


class FakeTokenizer:
    def __init__(self):
        self.stoi = {
            "<bos>": 0,
            "<user>": 1,
            "<sep>": 2,
            "<assistant>": 3,
            "<eos>": 4,
            "<pad>": 5,
            "<hi>": 6,
            "<en>": 7,
            "<mix>": 8,
        }
        self.pad = self.stoi["<pad>"]

    def encode(self, text):
        return [10 + ord(ch) % 40 for ch in text]


def test_sft_example_masks_prompt_and_trains_answer_plus_eos():
    tok = FakeTokenizer()
    example = build_example(
        {"user": "नमस्ते", "assistant": "नमस्ते दुनिया"},
        tok,
        max_seq_len=64,
        max_answer_tokens=32,
    )
    assert example is not None
    prefix_length = 2 + len(tok.encode("नमस्ते")) + 3
    assert example["tokens"][:2] == [tok.stoi["<bos>"], tok.stoi["<user>"]]
    assert example["tokens"][prefix_length - 1] == tok.stoi["<hi>"]
    assert example["tokens"][-1] == tok.stoi["<eos>"]
    assert example["loss_mask"][:prefix_length - 1] == [0] * (prefix_length - 1)
    assert example["loss_mask"][prefix_length - 1:] == [1] * (
        len(example["tokens"]) - prefix_length
    )


def test_sft_batch_padding_is_ignored_by_loss_mask():
    tok = FakeTokenizer()
    a = build_example({"user": "hello", "assistant": "world"}, tok, 64, 32)
    b = build_example({"user": "नमस्ते", "assistant": "उत्तर"}, tok, 64, 32)
    x, y, mask = make_batch([a, b], tok.pad)
    assert x.shape == y.shape == mask.shape
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    for i, row in enumerate((a, b)):
        n = len(row["tokens"]) - 1
        assert (x[i, :n] != tok.pad).all()
        assert (mask[i, :n] == row["loss_mask"]).all()
        assert (mask[i, n:] == 0).all()


def test_sft_truncates_prompt_before_dropping_answer_and_keeps_eos():
    tok = FakeTokenizer()
    example = build_example(
        {"user": "a" * 200, "assistant": "उत्तर"},
        tok,
        max_seq_len=32,
        max_answer_tokens=16,
    )
    assert example is not None
    assert len(example["tokens"]) <= 32
    assert example["tokens"][-1] == tok.stoi["<eos>"]
    assert sum(example["loss_mask"]) >= 2
