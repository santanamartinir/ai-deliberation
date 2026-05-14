"""
main.py

This is the main script that runs the deliberation experiment.

It:
- reads terminal arguments
- creates the local Ollama model
- chooses the protocol
- runs all rounds
- generates a final summary
- saves results to JSON and TXT
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from prompts import FINAL_MODERATOR_PROMPT
from protocols import run_json_call, run_round_direct, run_round_moderated, run_round_delphi
from metrics import compute_agreement, dominant_stance
from utils import slugify


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

DEFAULT_TOPIC = "Should universities integrate AI as a core learning tool?"


def parse_args():
    """
    Example:
    python main.py --model mistral --protocol moderated --topology star --rounds 3
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="mistral")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)

    parser.add_argument(
        "--protocol",
        type=str,
        choices=["direct", "moderated", "delphi"],
        default="moderated",
    )

    parser.add_argument(
        "--topology",
        type=str,
        choices=["fully_connected", "sequential", "star"],
        default="fully_connected",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create the local model
    llm = ChatOllama(model=args.model, temperature=args.temperature)

    agent_names = ["Agent A", "Agent B", "Agent C"]

    conversation_log = []
    round_summaries = []

    print(f"[START] {datetime.now().isoformat(timespec='seconds')}")
    print(f"[INFO] model={args.model}")
    print(f"[INFO] protocol={args.protocol}")
    print(f"[INFO] topology={args.topology}")
    print(f"[INFO] rounds={args.rounds}")
    print(f"[INFO] topic={args.topic}")

    previous_moderator_summary = None

    # Run all rounds
    for round_id in range(1, args.rounds + 1):
        print(f"[ROUND] {round_id}/{args.rounds}")

        if args.protocol == "direct":
            round_outputs = run_round_direct(
                llm=llm,
                topic=args.topic,
                round_id=round_id,
                topology=args.topology,
                agent_names=agent_names,
                conversation_log=conversation_log,
            )

            conversation_log.extend(round_outputs)

            agreement_pct, disagreement_pct = compute_agreement(round_outputs)
            dom_stance = dominant_stance(round_outputs)

            round_summaries.append({
                "round": round_id,
                "agreement_pct": agreement_pct,
                "disagreement_pct": disagreement_pct,
                "dominant_stance": dom_stance,
                "moderator_summary": "No moderator in direct protocol.",
                "main_agreements": [],
                "main_disagreements": [],
                "dominant_theme": "N/A",
            })

        elif args.protocol == "moderated":
            round_outputs, summary = run_round_moderated(
                llm=llm,
                topic=args.topic,
                round_id=round_id,
                topology=args.topology,
                agent_names=agent_names,
                conversation_log=conversation_log,
                previous_moderator_summary=previous_moderator_summary,
            )

            conversation_log.extend(round_outputs)
            round_summaries.append(summary)
            previous_moderator_summary = summary["moderator_summary"]

        elif args.protocol == "delphi":
            round_outputs, summary = run_round_delphi(
                llm=llm,
                topic=args.topic,
                round_id=round_id,
                agent_names=agent_names,
                previous_round_summary=previous_moderator_summary,
            )

            conversation_log.extend(round_outputs)
            round_summaries.append(summary)
            previous_moderator_summary = summary["moderator_summary"]

        latest_summary = round_summaries[-1]
        print(
            f'[ROUND SUMMARY] agreement={latest_summary["agreement_pct"]}% | '
            f'disagreement={latest_summary["disagreement_pct"]}% | '
            f'dominant={latest_summary["dominant_stance"]}'
        )

    # Generate final summary
    final_input = {
        "topic": args.topic,
        "protocol": args.protocol,
        "topology": args.topology,
        "round_summaries": round_summaries,
        "conversation_log": conversation_log,
    }

    final_mod_raw = run_json_call(
        llm,
        FINAL_MODERATOR_PROMPT,
        f"Here is the full discussion data:\n{json.dumps(final_input, indent=2, ensure_ascii=False)}",
    )

    final_summary = {
        "final_summary": str(final_mod_raw.get("final_summary", "")).strip(),
        "consensus_assessment": str(final_mod_raw.get("consensus_assessment", "")).strip(),
        "strongest_points": final_mod_raw.get("strongest_points", []),
    }

    # Build file name
    base_name = (
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        f"_{args.model}_{args.protocol}_{args.topology}_{slugify(args.topic)}"
    )

    result = {
        "config": {
            "model": args.model,
            "temperature": args.temperature,
            "rounds": args.rounds,
            "protocol": args.protocol,
            "topology": args.topology,
            "topic": args.topic,
        },
        "conversation_log": conversation_log,
        "round_summaries": round_summaries,
        "final_summary": final_summary,
    }

    # Save JSON
    json_path = LOG_DIR / f"{base_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Save TXT
    txt_path = LOG_DIR / f"{base_name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        cfg = result["config"]

        f.write("--- CONFIG ---\n")
        for key, value in cfg.items():
            f.write(f"{key}: {value}\n")

        f.write("\n--- CONVERSATION ---\n")
        for item in conversation_log:
            f.write(
                f'\nRound {item["round"]} | {item["speaker"]} '
                f'| stance={item["stance"]} | confidence={item["confidence"]}\n'
            )
            f.write(f'{item["message"]}\n')

        f.write("\n--- ROUND SUMMARIES ---\n")
        for item in round_summaries:
            f.write(
                f'\nRound {item["round"]}: '
                f'agreement={item["agreement_pct"]}% | '
                f'disagreement={item["disagreement_pct"]}% | '
                f'dominant={item["dominant_stance"]}\n'
            )
            f.write(f'Moderator summary: {item["moderator_summary"]}\n')
            f.write(f'Agreements: {item["main_agreements"]}\n')
            f.write(f'Disagreements: {item["main_disagreements"]}\n')
            f.write(f'Dominant theme: {item["dominant_theme"]}\n')

        f.write("\n--- FINAL SUMMARY ---\n")
        f.write(f'{final_summary["final_summary"]}\n')
        f.write(f'Consensus assessment: {final_summary["consensus_assessment"]}\n')
        f.write(f'Strongest points: {final_summary["strongest_points"]}\n')

    print("[DONE] Experiment finished.")
    print(f"[SAVED] JSON log: {json_path}")
    print(f"[SAVED] TXT log:  {txt_path}")


if __name__ == "__main__":
    main()