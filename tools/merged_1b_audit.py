from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bpe_tokenizer import BPETokenizer
from moe_model import MoEConfig, MoELM


MERGE_RUN_ID = 42342513329
MERGE_ARTIFACT_ID = 13562835269
MERGE_DIGEST = "sha256:012580196fe2f47ea9e9b4e2a7b087525b3dd770c00a9a6b85ef72766348d714"
SOURCE_RUNS = {
    "copy_0": 42275510662,
    "copy_1": 42273920152,
    "copy_2": 42271805121,
    "copy_3": 42267619625,
    "copy_4": 42272958418,
}
PROMPTS = [
    "Cryptography protects data by converting plaintext into",
    "A network firewall examines packets to decide",
    "SQL injection happens when untrusted input is inserted into",
    "An incident response plan should begin with",
    "Access control ensures that each user can",
    "साइबर सुरक्षा में नेटवर्क ट्रैफिक का विश्लेषण",
]


def find_one(root: Path, name: str) -> Path:
    found = list(root.rglob(name))
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one {name}, found {len(found)}")
    return found[0]


def repeated_ngram_ratio(tokens: list[int], n: int = 3) -> float:
    grams = [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def main() -> int:
    root = Path("artifact")
    report_dir = Path("audit")
    report_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "experiment": "read-only merged-1b structural and inference audit",
        "training_performed": False,
        "balanced_routing_enabled_for_audit": True,
        "source": {
            "merge_run_id": MERGE_RUN_ID,
            "artifact_id": MERGE_ARTIFACT_ID,
            "artifact_digest": MERGE_DIGEST,
            "input_checkpoint_runs": SOURCE_RUNS,
        },
        "runtime": {"torch": torch.__version__, "threads": 2},
        "checks": {},
        "findings": [
            {
                "severity": "high",
                "area": "router_initialization",
                "finding": (
                    "merge_experts.py appends repeated copies of old_router.mean() "
                    "as the router row for every newly added expert. Within each "
                    "22-expert incoming group, those rows are identical; top-2 "
                    "routing may therefore starve most of the added experts. "
                    "The measured routing counts below determine whether this "
                    "occurs on this checkpoint."
                ),
            },
            {
                "severity": "medium",
                "area": "shared_weight_averaging",
                "finding": (
                    "The merge workflow folds five checkpoints sequentially and "
                    "averages shared tensors as (accumulator + incoming) / 2 each "
                    "time. This is order-biased, not a five-way mean: the final "
                    "checkpoint receives 50% weight, while earlier copies receive "
                    "progressively less."
                ),
            },
            {
                "severity": "medium",
                "area": "expert_deduplication",
                "finding": (
                    "Duplicate detection uses cosine similarity of concatenated "
                    "raw expert weights, not measured functional equivalence. "
                    "The merge log reports 22 additions and zero removals at "
                    "threshold 0.999 for each incoming copy."
                ),
            },
            {
                "severity": "info",
                "area": "task_scope",
                "finding": (
                    "The source training workflow uses vocabulary warm-up followed "
                    "by raw security-domain text continuation. This audit tests "
                    "continuation and routing, not instruction-following or chat "
                    "quality."
                ),
            },
        ],
    }

    try:
        ckpt_path = find_one(root, "merged.pt")
        tokenizer_path = find_one(root, "tokenizer.json")
        summary_path = find_one(root, "summary.json")
        report["files"] = {
            "merged_checkpoint_bytes": ckpt_path.stat().st_size,
            "tokenizer_bytes": tokenizer_path.stat().st_size,
            "summary_bytes": summary_path.stat().st_size,
        }
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report["artifact_summary"] = summary

        # Safe, memory-mapped tensor-only load; avoids duplicating ~4 GB weights.
        ck = torch.load(
            ckpt_path, map_location="cpu", mmap=True, weights_only=True
        )
        cfg_data = dict(ck["config"])
        cfg_data["balanced_routing"] = True
        cfg = MoEConfig(**cfg_data)
        state = ck["model"]
        counts = cfg.counts()
        report["architecture"] = {
            "vocab": cfg.vocab,
            "dim": cfg.dim,
            "layers": cfg.layers,
            "heads": cfg.heads,
            "seq_len": cfg.seq_len,
            "experts_per_layer": cfg.experts,
            "top_k": cfg.top_k,
            "ffn_mult": cfg.ffn_mult,
            "total_parameters_from_config": counts["total"],
            "active_parameters_per_token": counts["active"],
            "active_fraction": round(counts["active"] / counts["total"], 6),
        }

        expected = {
            "parameters": 1_046_403_840,
            "active_parameters": 27_088_896,
            "experts": 110,
            "layers": 8,
            "dim": 384,
            "vocab": 8192,
        }
        observed = {
            "parameters": counts["total"],
            "active_parameters": counts["active"],
            "experts": cfg.experts,
            "layers": cfg.layers,
            "dim": cfg.dim,
            "vocab": cfg.vocab,
        }
        report["checks"]["architecture_matches_merge_log"] = {
            "pass": observed == expected,
            "expected": expected,
            "observed": observed,
        }

        # Validate all tensor payloads for finiteness and all expert/router shapes.
        nonfinite = []
        for key, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                nonfinite.append(f"{key}:not_tensor")
            elif tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                nonfinite.append(key)
        layer_shapes = []
        pattern = re.compile(r"blocks\.(\d+)\.moe\.experts\.(\d+)\.w1\.weight$")
        for layer in range(cfg.layers):
            w1_ids = sorted(
                int(m.group(2)) for key in state
                if (m := pattern.match(key)) and int(m.group(1)) == layer
            )
            router = state.get(f"blocks.{layer}.moe.router.weight")
            complete = w1_ids == list(range(cfg.experts))
            router_ok = (
                isinstance(router, torch.Tensor)
                and tuple(router.shape) == (cfg.experts, cfg.dim)
            )
            layer_shapes.append({
                "layer": layer,
                "expert_count": len(w1_ids),
                "indices_contiguous": complete,
                "router_shape": None if router is None else list(router.shape),
                "router_shape_valid": router_ok,
            })
        finite_ok = not nonfinite
        shape_ok = all(x["indices_contiguous"] and x["router_shape_valid"]
                       for x in layer_shapes)
        report["checks"]["all_checkpoint_tensors_finite"] = {
            "pass": finite_ok,
            "nonfinite_or_invalid_tensor_count": len(nonfinite),
            "examples": nonfinite[:10],
        }
        report["checks"]["expert_and_router_shapes"] = {
            "pass": shape_ok,
            "per_layer": layer_shapes,
        }

        # Audit repeated router rows introduced by the merge.
        router_rows = []
        for layer in range(cfg.layers):
            r = state[f"blocks.{layer}.moe.router.weight"]
            groups = []
            group_means = []
            for group in range(5):
                slab = r[group * 22:(group + 1) * 22].float()
                within = float((slab - slab[:1]).abs().max().item())
                mean = slab.mean(0)
                group_means.append(mean)
                groups.append({
                    "copy_group": group,
                    "max_abs_delta_from_first_row": within,
                    "all_rows_exactly_equal": bool(torch.equal(
                        slab, slab[:1].expand_as(slab)
                    )),
                })
            new_group_distances = []
            for i in range(1, 5):
                for j in range(i + 1, 5):
                    new_group_distances.append(float(
                        (group_means[i] - group_means[j]).abs().max().item()
                    ))
            router_rows.append({
                "layer": layer,
                "groups": groups,
                "max_abs_delta_between_new_group_means": (
                    max(new_group_distances) if new_group_distances else 0.0
                ),
            })
        report["router_row_audit"] = router_rows

        # Construct on meta and assign memory-mapped CPU tensors. Re-tie the
        # output head exactly as the architecture specifies.
        tie_equal_in_checkpoint = torch.equal(
            state["embed.weight"], state["head.weight"]
        )
        with torch.device("meta"):
            model = MoELM(cfg)
        load_result = model.load_state_dict(state, strict=True, assign=True)
        model.head.weight = model.embed.weight
        model.eval()
        actual_params = sum(p.numel() for p in model.parameters())
        report["checks"]["strict_model_load"] = {
            "pass": not load_result.missing_keys and not load_result.unexpected_keys,
            "missing_keys": load_result.missing_keys,
            "unexpected_keys": load_result.unexpected_keys,
        }
        report["checks"]["tied_embedding_head"] = {
            "pass": tie_equal_in_checkpoint and model.head.weight is model.embed.weight,
            "checkpoint_tensors_equal": tie_equal_in_checkpoint,
            "model_parameter_is_shared": model.head.weight is model.embed.weight,
        }
        report["checks"]["unique_parameter_count"] = {
            "pass": actual_params == counts["total"],
            "observed": actual_params,
            "expected": counts["total"],
        }

        tokenizer = BPETokenizer.load(tokenizer_path)
        roundtrips = []
        for sample in [
            "Cryptography and network security",
            "साइबर सुरक्षा में नेटवर्क ट्रैफिक का विश्लेषण",
            "SQL injection occurs when input is not validated.",
        ]:
            encoded = tokenizer.encode(sample)
            decoded = tokenizer.decode(encoded)
            roundtrips.append({
                "text": sample,
                "token_count": len(encoded),
                "exact_roundtrip": decoded == sample,
            })
        report["tokenizer"] = {
            "size": tokenizer.size,
            "roundtrip_probes": roundtrips,
            "all_roundtrips_exact": all(x["exact_roundtrip"] for x in roundtrips),
        }

        # Explicit batch-coverage probe: 55 prompt rows x 32 positions x top-2
        # = 3,520 assignments per layer, enough to give every one of 110 experts
        # work. The 55 rows are a repeated, mixed Hindi/English security prompt
        # set; this is a routing-coverage test, not a language-quality metric.
        coverage_ids = []
        for i in range(55):
            ids = tokenizer.encode(PROMPTS[i % len(PROMPTS)])[:32]
            ids = ids + [tokenizer.eos] * (32 - len(ids))
            coverage_ids.append(ids)
        coverage_batch = torch.tensor(coverage_ids, dtype=torch.long)
        coverage_usage = []
        torch.set_num_threads(2)
        with torch.inference_mode():
            coverage_logits = model(coverage_batch)
            if not bool(torch.isfinite(coverage_logits).all()):
                raise RuntimeError("balanced coverage batch produced non-finite logits")
            for li, block in enumerate(model.blocks):
                usage = block["moe"].last_usage.detach().cpu().tolist()
                coverage_usage.append({
                    "layer": li,
                    "active_experts": sum(n > 0 for n in usage),
                    "assignments": int(sum(usage)),
                    "min_assignments_per_expert": int(min(usage)),
                    "max_assignments_per_expert": int(max(usage)),
                    "per_expert_assignments": [int(n) for n in usage],
                })
        report["balanced_batch_coverage"] = {
            "batch_sequences": 55,
            "sequence_length": 32,
            "top_k": cfg.top_k,
            "expected_assignments_per_layer": 55 * 32 * cfg.top_k,
            "all_layers_use_all_experts": all(
                x["active_experts"] == cfg.experts for x in coverage_usage
            ),
            "per_layer": coverage_usage,
        }
        if not report["balanced_batch_coverage"]["all_layers_use_all_experts"]:
            raise RuntimeError("balanced routing failed full 110-expert batch coverage")

        # Single-prompt greedy continuations remain ordinary top-k because a
        # one-sequence batch is too small to cover 110 experts at top-2.
        usage_by_layer = [[0] * cfg.experts for _ in range(cfg.layers)]
        generations = []
        with torch.inference_mode():
            for prompt in PROMPTS:
                ids = tokenizer.encode(prompt)
                if not ids:
                    generations.append({
                        "prompt": prompt, "error": "tokenizer produced no ids"
                    })
                    continue
                ids = ids[-cfg.seq_len:]
                x = torch.tensor([ids], dtype=torch.long)
                logits = model(x)
                logits_finite = bool(torch.isfinite(logits).all())
                if tuple(logits.shape) != (1, len(ids), cfg.vocab):
                    raise RuntimeError(f"unexpected logits shape {tuple(logits.shape)}")
                for li, block in enumerate(model.blocks):
                    counts_now = block["moe"].last_usage.detach().cpu().tolist()
                    usage_by_layer[li] = [
                        a + int(b) for a, b in zip(usage_by_layer[li], counts_now)
                    ]

                # Greedy raw continuation; no chat wrapper is added.
                context = list(ids)
                out_ids = []
                for _ in range(24):
                    xx = torch.tensor([context[-cfg.seq_len:]], dtype=torch.long)
                    next_logits = model(xx)[0, -1, :]
                    nxt = int(torch.argmax(next_logits).item())
                    if nxt == tokenizer.eos:
                        break
                    out_ids.append(nxt)
                    context.append(nxt)
                continuation = tokenizer.decode(out_ids)
                generations.append({
                    "prompt": prompt,
                    "continuation": continuation,
                    "generated_tokens": len(out_ids),
                    "unique_token_ratio": (
                        round(len(set(out_ids)) / len(out_ids), 4)
                        if out_ids else 0.0
                    ),
                    "repeated_trigram_ratio": round(
                        repeated_ngram_ratio(out_ids, 3), 4
                    ),
                    "initial_logits_finite": logits_finite,
                })

        layer_usage = []
        all_usage = [0] * cfg.experts
        for li, usage in enumerate(usage_by_layer):
            for i, n in enumerate(usage):
                all_usage[i] += n
            layer_usage.append({
                "layer": li,
                "active_experts": sum(n > 0 for n in usage),
                "assignments": sum(usage),
                "copy_groups": [
                    {
                        "copy_group": g,
                        "assignments": sum(usage[g*22:(g+1)*22]),
                        "active_experts": sum(
                            n > 0 for n in usage[g*22:(g+1)*22]
                        ),
                    }
                    for g in range(5)
                ],
            })
        group_assignments = [
            sum(all_usage[g*22:(g+1)*22]) for g in range(5)
        ]
        group_active = [
            sum(n > 0 for n in all_usage[g*22:(g+1)*22]) for g in range(5)
        ]
        total_assignments = sum(all_usage)
        report["routing"] = {
            "measurement": "prompt-prefix routing only; generation routing excluded",
            "total_assignments_across_layers": total_assignments,
            "unique_experts_used_across_all_layers": sum(n > 0 for n in all_usage),
            "copy_groups": [
                {
                    "copy_group": g,
                    "available_expert_slots": 22 * cfg.layers,
                    "unique_experts_used": group_active[g],
                    "token_assignments": group_assignments[g],
                    "assignment_share": (
                        round(group_assignments[g] / total_assignments, 6)
                        if total_assignments else 0.0
                    ),
                }
                for g in range(5)
            ],
            "per_layer": layer_usage,
        }
        report["generations"] = generations

        structural_pass = all([
            report["checks"]["architecture_matches_merge_log"]["pass"],
            report["checks"]["all_checkpoint_tensors_finite"]["pass"],
            report["checks"]["expert_and_router_shapes"]["pass"],
            report["checks"]["strict_model_load"]["pass"],
            report["checks"]["tied_embedding_head"]["pass"],
            report["checks"]["unique_parameter_count"]["pass"],
        ])
        generation_nonempty = all(
            row.get("generated_tokens", 0) > 0 and row.get("continuation", "").strip()
            for row in generations
        )
        report["generation_smoke_test"] = {
            "pass": generation_nonempty,
            "nonempty_continuations": sum(
                bool(row.get("continuation", "").strip()) for row in generations
            ),
            "prompt_count": len(generations),
            "quality_certified": False,
            "note": "Raw continuation only; instruction following and factuality require separate evaluation.",
        }
        report["result"] = {
            "structural_and_forward_pass": structural_pass,
            "generation_smoke_test_pass": generation_nonempty,
            "chat_quality_certified": False,
            "overall_pass": structural_pass and generation_nonempty,
            "failed_structural_checks": [
                name for name, item in report["checks"].items()
                if isinstance(item, dict) and item.get("pass") is False
            ],
            "explanation": "Structural integrity and generation quality are separate results.",
        }
        (report_dir / "merged_1b_audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report["result"], ensure_ascii=False, indent=2))
        print("Routing summary:", json.dumps(report["routing"]["copy_groups"]))
        print("Continuations:")
        for row in generations:
            print(f"PROMPT: {row['prompt']}\nCONTINUATION: {row.get('continuation', row.get('error'))}\n")
        return 0 if structural_pass else 2

    except Exception as exc:
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
        (report_dir / "merged_1b_audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
