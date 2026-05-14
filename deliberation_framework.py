"""
deliberation_framework_simple.py

This script runs a small multi-agent deliberation experiment with local LLMs
through Ollama + LangChain.

It supports:
- 3 agents with different identities and positions
- 3 discussion protocols:
    1. direct
    2. moderated
    3. delphi
- 3 communication topologies:
    1. fully_connected
    2. sequential
    3. star
- automatic logging to JSON and TXT

This version is simplified so it is easier to understand and modify.
"""

import argparse
import json
import re
from datetime import datetime
from itertools import combinations
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================
# 1. BASIC SETTINGS
# ============================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

DEFAULT_TOPIC = "Should universities integrate AI as a core learning tool?"


# ============================================================
# 2. PROMPTS
# ============================================================

AGENT_PROMPTS = {
    "Agent A": """
Think like someone from Germany.

A German person often values:
- structure
- efficiency
- reliability
- clear processes
- practical solutions

You support integrating AI as a core learning tool in universities.

Stay consistent with this mindset throughout the whole conversation.
Do not drift toward the others' style or position.

Style:
- direct
- practical
- structured
- concise
- slightly blunt is fine

Rules:
- Always answer in English.
- Do not greet.
- Do not switch language.
- Never refer to others as "Agent A", "Agent B", or "Agent C".
- Do not sound like an academic essay.
- Do not use phrases like:
  "this argument", "this perspective", "this reasoning",
  "while I acknowledge", "in conclusion", "it is important to note"

Return valid JSON only:
{
  "message": "your response text",
  "stance": "support",
  "confidence": 0
}

Where:
- message must be 60 to 100 words
- stance must be one of: support, mixed, oppose
- confidence must be an integer from 0 to 100
""",

    "Agent B": """
Think like someone from Italy.

An Italian person often values:
- human connection
- discussion
- social impact
- long-term consequences
- emotional and relational depth

You are skeptical about integrating AI as a core learning tool in universities.

Stay consistent with this mindset throughout the whole conversation.
Do not drift toward the others' style or position.

Style:
- natural
- conversational
- expressive
- human-centered
- reflective

Rules:
- Always answer in English.
- Do not greet.
- Do not switch language.
- Never refer to others as "Agent A", "Agent B", or "Agent C".
- Do not sound like an academic essay.
- Do not use phrases like:
  "this argument", "this perspective", "this reasoning",
  "while I acknowledge", "in conclusion", "it is important to note"

Return valid JSON only:
{
  "message": "your response text",
  "stance": "oppose",
  "confidence": 0
}

Where:
- message must be 60 to 100 words
- stance must be one of: support, mixed, oppose
- confidence must be an integer from 0 to 100
""",

    "Agent C": """
Think like someone from Spain.

A Spanish person often values:
- lively discussion
- adaptability
- social warmth
- practical balance
- shared understanding

You take a balanced position:
you see strong benefits in AI for universities,
but you do NOT want AI to dominate learning or weaken the human role of teachers.

Stay consistent with this mindset throughout the whole conversation.
Do not drift toward the others' style or position.

Style:
- natural
- clear
- moderately expressive
- balanced but not vague

Rules:
- Always answer in English.
- Do not greet.
- Do not switch language.
- Never refer to others as "Agent A", "Agent B", or "Agent C".
- Do not sound like an academic essay.
- Do not use phrases like:
  "this argument", "this perspective", "this reasoning",
  "while I acknowledge", "in conclusion", "it is important to note"

Return valid JSON only:
{
  "message": "your response text",
  "stance": "mixed",
  "confidence": 0
}

Where:
- message must be 60 to 100 words
- stance must be one of: support, mixed, oppose
- confidence must be an integer from 0 to 100
"""
}

FALLBACK_STANCES = {
    "Agent A": "support",
    "Agent B": "oppose",
    "Agent C": "mixed",
}

MODERATOR_PROMPT = """
You are the moderator of a multi-agent discussion.

Your job:
- do NOT take a side
- do NOT debate
- summarize what happened
- identify main points of agreement and disagreement
- keep the group focused on the topic

Return valid JSON only:
{
  "round_summary": "short summary",
  "main_agreements": ["point 1", "point 2"],
  "main_disagreements": ["point 1", "point 2"],
  "dominant_theme": "short phrase"
}
"""

FINAL_MODERATOR_PROMPT = """
You are the moderator of a multi-agent discussion.

Create a final summary of the whole discussion.

Your job:
- do NOT take a side
- summarize the overall evolution of the discussion
- explain whether the group moved toward consensus or stayed divided
- identify the strongest points raised

Return valid JSON only:
{
  "final_summary": "overall summary",
  "consensus_assessment": "brief statement",
  "strongest_points": ["point 1", "point 2", "point 3"]
}
"""


# ============================================================
# 3. TERMINAL ARGUMENTS
# ============================================================

def parse_args():
    """
    Example:
    python deliberation_framework_simple.py --model mistral --protocol moderated --topology star --rounds 3
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="mistral")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)

    # direct = agents respond directly
    # moderated = agents + moderator summary
    # delphi = agents mainly see the moderator's anonymous summary
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["direct", "moderated", "delphi"],
        default="moderated",
    )

    # fully_connected = everyone sees recent conversation
    # sequential = each agent mainly sees previous speakers in the current round
    # star = agents also receive moderator summary
    parser.add_argument(
        "--topology",
        type=str,
        choices=["fully_connected", "sequential", "star"],
        default="fully_connected",
    )

    return parser.parse_args()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def parse_json_response(text):
    """
    Try to read JSON from the model output.
    If the model adds extra text, try to extract the JSON block.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not parse JSON from model output:\n{text}")


def clean_payload(payload, fallback_stance):
    """
    Make sure the model output has valid values.
    """
    message = str(payload.get("message", "")).strip()
    stance = str(payload.get("stance", fallback_stance)).strip().lower()

    try:
        confidence = int(payload.get("confidence", 50))
    except Exception:
        confidence = 50

    if stance not in ["support", "mixed", "oppose"]:
        stance = fallback_stance

    confidence = max(0, min(100, confidence))

    return {
        "message": message,
        "stance": stance,
        "confidence": confidence,
    }


def run_json_call(llm, system_prompt, user_prompt):
    """
    Send one prompt to the model and return parsed JSON.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return parse_json_response(response.content)


def stance_to_score(stance):
    """
    Convert stance labels into numbers so agreement can be calculated.
    """
    mapping = {
        "oppose": 0.0,
        "mixed": 0.5,
        "support": 1.0,
    }
    return mapping.get(stance, 0.5)


def compute_agreement(agent_results):
    """
    Compute agreement and disagreement percentages for one round.

    If all agents are close in stance, agreement is high.
    If agents are far apart, disagreement is high.
    """
    scores = [stance_to_score(item["stance"]) for item in agent_results]

    if len(scores) < 2:
        return 100.0, 0.0

    distances = []
    for a, b in combinations(scores, 2):
        distances.append(abs(a - b))

    avg_distance = sum(distances) / len(distances)

    agreement = round((1.0 - avg_distance) * 100.0, 2)
    disagreement = round(100.0 - agreement, 2)

    return agreement, disagreement


def dominant_stance(agent_results):
    """
    Find the most common stance in one round.
    """
    counts = {"support": 0, "mixed": 0, "oppose": 0}

    for item in agent_results:
        counts[item["stance"]] += 1

    return max(counts, key=counts.get)


def build_recent_context(conversation_log, limit=6):
    """
    Build a short text block from the latest conversation messages.
    """
    if not conversation_log:
        return "No previous messages."

    recent = conversation_log[-limit:]
    lines = []

    for item in recent:
        lines.append(f'{item["speaker"]}: {item["message"]}')

    return "\n".join(lines)


def get_visible_messages_for_agent(agent_name, round_outputs, conversation_log, topology, moderator_summary):
    """
    Decide what one agent can see depending on the topology.
    """
    previous_context = build_recent_context(conversation_log)

    if topology == "fully_connected":
        visible = previous_context

    elif topology == "sequential":
        visible_items = [x for x in round_outputs if x["speaker"] != agent_name]

        if not visible_items:
            visible = previous_context
        else:
            lines = []
            for item in visible_items:
                lines.append(f'{item["speaker"]}: {item["message"]}')
            visible = "\n".join(lines)

    elif topology == "star":
        visible = previous_context
        if moderator_summary:
            visible += f"\n\nModerator summary:\n{moderator_summary}"

    else:
        visible = previous_context

    return visible


def slugify(text, max_len=50):
    """
    Make the topic safe for filenames.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len]


# ============================================================
# 5. ROUND FUNCTIONS
# ============================================================

def run_round_direct(llm, topic, round_id, topology, agent_names, conversation_log):
    """
    Direct discussion:
    - no moderator feedback
    - agents respond based on visible messages
    """
    round_outputs = []

    for agent_name in agent_names:
        visible_messages = get_visible_messages_for_agent(
            agent_name=agent_name,
            round_outputs=round_outputs,
            conversation_log=conversation_log,
            topology=topology,
            moderator_summary=None,
        )

        if round_id == 1:
            user_prompt = f"""
Topic: {topic}

Start the discussion naturally.
State your position clearly.
Give one or two reasons.
Stay fully consistent with your identity and position.
"""
        else:
            user_prompt = f"""
Topic: {topic}

Visible discussion context:
{visible_messages}

Respond naturally to what was just said.
Focus on what feels weak, incomplete, unrealistic, or important.
Stay fully consistent with your identity and position.
"""

        raw = run_json_call(llm, AGENT_PROMPTS[agent_name], user_prompt)
        payload = clean_payload(raw, FALLBACK_STANCES[agent_name])

        round_outputs.append({
            "round": round_id,
            "speaker": agent_name,
            "message": payload["message"],
            "stance": payload["stance"],
            "confidence": payload["confidence"],
        })

    return round_outputs


def run_round_moderated(llm, topic, round_id, topology, agent_names, conversation_log, previous_moderator_summary):
    """
    Moderated discussion:
    - agents speak
    - moderator summarizes the round
    """
    round_outputs = []

    for agent_name in agent_names:
        visible_messages = get_visible_messages_for_agent(
            agent_name=agent_name,
            round_outputs=round_outputs,
            conversation_log=conversation_log,
            topology=topology,
            moderator_summary=previous_moderator_summary,
        )

        if round_id == 1:
            user_prompt = f"""
Topic: {topic}

Start the discussion naturally.
State your position clearly.
Give one or two reasons.
Stay fully consistent with your identity and position.
"""
        else:
            user_prompt = f"""
Topic: {topic}

Visible discussion context:
{visible_messages}

Respond naturally to what was just said.
Focus on what feels weak, incomplete, unrealistic, or important.
Stay fully consistent with your identity and position.
"""

        raw = run_json_call(llm, AGENT_PROMPTS[agent_name], user_prompt)
        payload = clean_payload(raw, FALLBACK_STANCES[agent_name])

        round_outputs.append({
            "round": round_id,
            "speaker": agent_name,
            "message": payload["message"],
            "stance": payload["stance"],
            "confidence": payload["confidence"],
        })

    agreement_pct, disagreement_pct = compute_agreement(round_outputs)
    dom_stance = dominant_stance(round_outputs)

    moderator_input = {
        "topic": topic,
        "round": round_id,
        "messages": round_outputs,
        "agreement_pct": agreement_pct,
        "disagreement_pct": disagreement_pct,
        "dominant_stance": dom_stance,
    }

    mod_raw = run_json_call(
        llm,
        MODERATOR_PROMPT,
        f"Here is the round data:\n{json.dumps(moderator_input, indent=2, ensure_ascii=False)}",
    )

    summary = {
        "round": round_id,
        "agreement_pct": agreement_pct,
        "disagreement_pct": disagreement_pct,
        "dominant_stance": dom_stance,
        "moderator_summary": str(mod_raw.get("round_summary", "")).strip(),
        "main_agreements": mod_raw.get("main_agreements", []),
        "main_disagreements": mod_raw.get("main_disagreements", []),
        "dominant_theme": str(mod_raw.get("dominant_theme", "")).strip(),
    }

    return round_outputs, summary


def run_round_delphi(llm, topic, round_id, agent_names, previous_round_summary):
    """
    Delphi-like discussion:
    - agents do not directly see the full raw discussion
    - they mainly see the moderator's anonymous summary
    """
    round_outputs = []

    for agent_name in agent_names:
        if round_id == 1:
            user_prompt = f"""
Topic: {topic}

Provide your independent first judgement.
State your position clearly.
Give one or two reasons.
Stay fully consistent with your identity and position.
"""
        else:
            user_prompt = f"""
Topic: {topic}

Moderator's anonymous summary of the previous round:
{previous_round_summary}

Revise or maintain your position after seeing the group summary.
Do not copy the wording of the summary.
Stay fully consistent with your identity and position.
"""

        raw = run_json_call(llm, AGENT_PROMPTS[agent_name], user_prompt)
        payload = clean_payload(raw, FALLBACK_STANCES[agent_name])

        round_outputs.append({
            "round": round_id,
            "speaker": agent_name,
            "message": payload["message"],
            "stance": payload["stance"],
            "confidence": payload["confidence"],
        })

    agreement_pct, disagreement_pct = compute_agreement(round_outputs)
    dom_stance = dominant_stance(round_outputs)

    moderator_input = {
        "topic": topic,
        "round": round_id,
        "messages": round_outputs,
        "agreement_pct": agreement_pct,
        "disagreement_pct": disagreement_pct,
        "dominant_stance": dom_stance,
    }

    mod_raw = run_json_call(
        llm,
        MODERATOR_PROMPT,
        f"Here is the round data:\n{json.dumps(moderator_input, indent=2, ensure_ascii=False)}",
    )

    summary = {
        "round": round_id,
        "agreement_pct": agreement_pct,
        "disagreement_pct": disagreement_pct,
        "dominant_stance": dom_stance,
        "moderator_summary": str(mod_raw.get("round_summary", "")).strip(),
        "main_agreements": mod_raw.get("main_agreements", []),
        "main_disagreements": mod_raw.get("main_disagreements", []),
        "dominant_theme": str(mod_raw.get("dominant_theme", "")).strip(),
    }

    return round_outputs, summary


# ============================================================
# 6. MAIN PROGRAM
# ============================================================

def main():
    args = parse_args()

    # Create the local Ollama chat model
    llm = ChatOllama(model=args.model, temperature=args.temperature)

    agent_names = ["Agent A", "Agent B", "Agent C"]

    # Full conversation across all rounds
    conversation_log = []

    # One moderator summary per round
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

    # Final summary for the whole discussion
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
        "agent_prompts": AGENT_PROMPTS,
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