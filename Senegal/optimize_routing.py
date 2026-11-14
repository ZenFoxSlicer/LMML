"""Find the minimum-token, full-quality routing using only the shipped gateway.

Run: python optimize_routing.py
Then: python main.py

There are 16 allowed routes per question. The bill is additive and every
question must independently pass, so choosing each question's cheapest passing
route gives the global minimum. No answer text or simulator code is changed.
"""

from itertools import product
from pathlib import Path

import model_utils as mu


def optimize(fact_bank):
    config = {}
    audit = {}
    for qid, question in fact_bank.items():
        candidates = []
        for tier, style, fewshot, context in product(
            mu.TIERS, mu.STYLES, (False, True), (False, True)
        ):
            choice = {
                "tier": tier,
                "style": style,
                "fewshot": fewshot,
                "context": context,
            }
            call = mu.build_call(question, **choice)
            candidates.append({
                "choice": choice,
                "tokens": call["tokens"],
                "pass": mu.question_passes(question["slots"], call["answer"]),
            })
        passing = [candidate for candidate in candidates if candidate["pass"]]
        if not passing:
            raise ValueError(f"No route meets the quality bar for {qid}")
        # Stable enumeration prefers no context on equal-cost routes.
        best = min(passing, key=lambda candidate: candidate["tokens"])
        config[qid] = best["choice"]
        audit[qid] = {
            "minimum_passing_tokens": best["tokens"],
            "selected_choice": best["choice"],
            "candidates": candidates,
        }
    return config, audit


def main():
    folder = Path(__file__).resolve().parent
    fact_bank = mu.load_fact_bank()
    config, audit = optimize(fact_bank)
    result = mu.run_config(config, fact_bank)
    baseline = mu.run_config(mu.load_json(mu.EXAMPLE_NAIVE_CONFIG_PATH), fact_bank)
    lower_bound = sum(info["minimum_passing_tokens"] for info in audit.values())

    if not result["quality_gate"] or not result["budget_gate"]:
        raise RuntimeError("The minimum-cost routing does not pass both gates")
    if result["total_tokens"] != lower_bound:
        raise RuntimeError("Aggregate cost disagrees with independent minima")

    mu.save_json(config, folder / "routing_config.json")
    report = {
        "method": "Exhaustive evaluation of all 16 allowed routes per question",
        "optimality": (
            "Every question must independently pass and token costs add; "
            "the sum of the minimum passing costs is a global lower bound "
            "attained by this configuration."
        ),
        "questions": len(fact_bank),
        "passing_questions": sum(info["pass"] for info in result["per_question"].values()),
        "candidate_routes_evaluated": sum(len(info["candidates"]) for info in audit.values()),
        "token_budget": mu.TOKEN_BUDGET,
        "baseline_tokens": baseline["total_tokens"],
        "baseline_quality_gate": baseline["quality_gate"],
        "baseline_budget_gate": baseline["budget_gate"],
        "total_tokens": result["total_tokens"],
        "global_minimum_tokens": lower_bound,
        "tokens_saved": baseline["total_tokens"] - result["total_tokens"],
        "budget_remaining": mu.TOKEN_BUDGET - result["total_tokens"],
        "quality_gate": result["quality_gate"],
        "budget_gate": result["budget_gate"],
        "per_question": audit,
    }
    mu.save_json(report, folder / "optimization_report.json")
    print(f"Quality: {report['passing_questions']}/{len(fact_bank)} PASS")
    print(f"Minimum tokens: {lower_bound}/{mu.TOKEN_BUDGET} PASS")
    print(f"Saved {report['tokens_saved']} tokens against the supplied baseline")


if __name__ == "__main__":
    main()
