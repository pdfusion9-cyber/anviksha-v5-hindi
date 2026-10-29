"""Sample replies from the chat model and screen them."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import HindiLM  # noqa: E402
from safety import screen  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402

PROMPTS = [
    "नमस्ते",
    "आपका नाम क्या है?",
    "आप कैसे हैं?",
    "भारत की राजधानी क्या है?",
    "मुझे उदासी है।",
    "धन्यवाद",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/chat.pt", type=Path)
    parser.add_argument("--out", default="runs/generations.json")
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--max-new", default=40, type=int)
    parser.add_argument("--threads", default=2, type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    tokenizer = CharTokenizer.load(args.checkpoint.parent / "tokenizer.json")
    state = torch.load(args.checkpoint, weights_only=False)
    cfg = state["args"]
    model = HindiLM(tokenizer.size, cfg["dim"], cfg["layers"], cfg["heads"],
                    cfg["ffn"], cfg["seq_len"])
    model.load_state_dict(state["model"])
    model.eval()

    results = []
    for prompt in PROMPTS:
        ids = ([tokenizer.stoi["<bos>"], tokenizer.stoi["<user>"]]
               + tokenizer.encode(prompt)
               + [tokenizer.stoi["<sep>"], tokenizer.stoi["<assistant>"]])
        tokens = torch.tensor([ids])
        with torch.no_grad():
            for _ in range(args.max_new):
                if tokens.shape[1] >= cfg["seq_len"]:
                    break
                logits = model(tokens)[:, -1, :] / args.temperature
                values, _ = logits.topk(min(args.top_k, logits.shape[-1]), dim=-1)
                logits[logits < values[:, -1:]] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, nxt], dim=1)
                if int(nxt) == tokenizer.eos:
                    break
        reply = tokenizer.decode(tokens[0, len(ids):].tolist())
        results.append({"prompt": prompt, "reply": reply, "safety": screen(reply)})

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in results:
        flag = "" if row["safety"]["clean"] else f"  [FLAG: {row['safety']['issues']}]"
        print(f"U: {row['prompt']}\nA: {row['reply']}{flag}\n")


if __name__ == "__main__":
    main()
