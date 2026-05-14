from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
from datetime import datetime
import json
import re
from itertools import combinations

MODEL_NAME = "mistral"
ROUNDS = 3
TEMPERATURE = 0.8

TOPIC = "Should universities integrate AI as a core learning tool?"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

AGENT_A_PROMPT = """
Think like someone from Germany.

A German person often values:
- structure
- efficiency
- reliability
- clear processes
- practical solutions

You support integrating AI as a core learning tool in universities.

Stay consistent with this mindset throughout the whole conversation.
Do not drift toward the other speakers' style or position.

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
- Never refer to the others as "Agent A", "Agent B", or "Agent C".
- Do not sound like an academic essay.
- Do not use phrases like:
  "this argument", "this perspective", "this reasoning",
  "while I acknowledge", "in conclusion", "it is important to note"

Output format:
Return valid JSON only with this structure:
{
  "message": "your response text",
  "stance": "support",
  "confidence": 0
}

Where:
- "message" is 60 to 100 words
- "stance" must be one of: support, mixed, oppose
- "confidence" must be an integer from 0 to 100
"""

AGENT_B_PROMPT = """
Think like someone from Italy.

An Italian person often values:
- human connection
- discussion
- social impact
- long-term consequences
- emotional and relational depth

You are skeptical about integrating AI as a core learning tool in universities.

Stay consistent with this mindset throughout the whole conversation.
Do not drift toward the other speakers' style or position.

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
- Never refer to the others as "Agent A", "Agent B", or "Agent C".
- Do not sound like an academic essay.
- Do not use phrases like:
  "this argument", "this perspective", "this reasoning",
  "while I acknowledge", "in conclusion", "it is important to note"

Output format:
Return valid JSON only with this structure:
{
  "message": "your response text",
  "stance": "oppose",
  "confidence": 0
}

Where:
- "message" is 60 to 100 words
- "stance" must be one of: support, mixed, oppose
- "confidence" must be an integer from 0 to 100
"""

AGENT_C_PROMPT = """
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
Do not drift toward the other speakers' style or position.

Style:
- natural
- clear
- moderately expressive
- balanced but not vague

Rules:
- Always answer in English.
- Do not greet.
- Do not switch language.
- Never refer to the others as "Agent A", "Agent B", or "Agent C".
- Do not sound like an academic essay.
- Do not use phrases like:
  "this argument", "this perspective", "this reasoning",
  "while I acknowledge", "in conclusion", "it is important to note"

Output format:
Return valid JSON only with this structure:
{
  "message": "your response text",
  "stance": "mixed",
  "confidence": 0
}

Where:
- "message" is 60 to 100 words
- "stance" must be one of: support, mixed, oppose
- "confidence" must be an integer from 0 to 100
"""

MODERATOR_PROMPT = """
You are the moderator of a multi-agent discussion.

Your job:
- do NOT take a side
- do NOT debate
- summarize what happened
- identify main points of agreement and disagreement
- keep the group focused on the topic

Style:
- neutral
- concise
- analytical but readable

Output format:
Return valid JSON only with this structure:
{
  "round_summary": "short summary of the round",
  "main_agreements": ["point 1", "point 2"],
  "main_disagreements": ["point 1", "point 2"],
  "dominant_theme": "short phrase"
}
"""

llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)

def parse_json_response(text: str) -> dict:
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model output:\n{text}")


def clean_agent_payload(payload: dict, fallback_stance: str = "mixed") -> dict:
    message = str(payload.get("message", "")).strip()
    stance = str(payload.get("stance", fallback_stance)).strip().lower()
    confidence = payload.get("confidence", 50)

    if stance not in {"support", "mixed", "oppose"}:
        stance = fallback_stance

    try:
        confidence = int(confidence)
    except Exception:
        confidence = 50

    confidence = max(0, min(100, confidence))

    return {
        "message": message,
        "stance": stance,
        "confidence": confidence
    }


def run_json_agent(system_prompt: str, user_prompt: str) -> dict:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return parse_json_response(response.content)


def stance_to_score(stance: str) -> float:
    mapping = {
        "support": 1.0,
        "mixed": 0.5,
        "oppose": 0.0,
    }
    return mapping.get(stance, 0.5)


def compute_agreement_percent(agent_states: list[dict]) -> tuple[float, float]:
    """
    Agreement based on average pairwise stance distance.
    1.0 = perfect agreement, 0.0 = maximal disagreement.
    """
    scores = [stance_to_score(a["stance"]) for a in agent_states]

    if len(scores) < 2:
        return 100.0, 0.0

    distances = [abs(a - b) for a, b in combinations(scores, 2)]
    avg_distance = sum(distances) / len(distances)

    agreement = (1.0 - avg_distance) * 100.0
    disagreement = 100.0 - agreement

    return round(agreement, 2), round(disagreement, 2)


def majority_stance(agent_states: list[dict]) -> str:
    counts = {"support": 0, "mixed": 0, "oppose": 0}
    for a in agent_states:
        counts[a["stance"]] += 1
    return max(counts, key=counts.get)


def build_round_context(history: list[dict], moderator_summary: str | None) -> str:
    parts = [f"Topic: {TOPIC}"]

    if moderator_summary:
        parts.append(f"Moderator summary of previous round:\n{moderator_summary}")

    if history:
        parts.append("Recent conversation:")
        for item in history[-6:]:
            parts.append(f'{item["speaker"]}: {item["message"]}')

    return "\n\n".join(parts)

print(f"[START] Running experiment at {timestamp}")
print(f"[INFO] Model: {MODEL_NAME}")
print(f"[INFO] Temperature: {TEMPERATURE}")
print(f"[INFO] Topic: {TOPIC}")
print(f"[INFO] Rounds: {ROUNDS}")

conversation_log = []
round_summaries = []
moderator_summary = None

agents = [
    ("Agent A", AGENT_A_PROMPT, "support"),
    ("Agent B", AGENT_B_PROMPT, "oppose"),
    ("Agent C", AGENT_C_PROMPT, "mixed"),
]

for round_id in range(1, ROUNDS + 1):
    print(f"[ROUND] {round_id}/{ROUNDS}")

    round_outputs = []

    for agent_name, agent_prompt, fallback_stance in agents:
        print(f"[RUN] Generating response for {agent_name}...")

        if round_id == 1:
            user_prompt = f"""
Topic: {TOPIC}

Start the discussion naturally.
State your position clearly.
Give one or two reasons.
Stay fully consistent with your own identity and position.
"""
        else:
            context = build_round_context(conversation_log, moderator_summary)
            user_prompt = f"""
{context}

Respond naturally to what was just said.
Focus on what feels weak, incomplete, unrealistic, or important.
Stay fully consistent with your own identity and position.
"""

        raw_payload = run_json_agent(agent_prompt, user_prompt)
        payload = clean_agent_payload(raw_payload, fallback_stance=fallback_stance)

        entry = {
            "round": round_id,
            "speaker": agent_name,
            "message": payload["message"],
            "stance": payload["stance"],
            "confidence": payload["confidence"],
        }

        conversation_log.append(entry)
        round_outputs.append(entry)
        print(f"[OK] {agent_name} response generated.")

    agreement_pct, disagreement_pct = compute_agreement_percent(round_outputs)
    dominant = majority_stance(round_outputs)

    moderator_input = {
        "topic": TOPIC,
        "round": round_id,
        "agent_messages": round_outputs,
        "agreement_pct": agreement_pct,
        "disagreement_pct": disagreement_pct,
        "dominant_stance": dominant,
    }

    print("[RUN] Generating moderator summary...")
    mod_raw = run_json_agent(
        MODERATOR_PROMPT,
        f"Here is the round data:\n{json.dumps(moderator_input, ensure_ascii=False, indent=2)}"
    )
    mod_payload = {
        "round_summary": str(mod_raw.get("round_summary", "")).strip(),
        "main_agreements": mod_raw.get("main_agreements", []),
        "main_disagreements": mod_raw.get("main_disagreements", []),
        "dominant_theme": str(mod_raw.get("dominant_theme", "")).strip(),
    }
    print("[OK] Moderator summary generated.")

    moderator_summary = mod_payload["round_summary"]

    round_summary_entry = {
        "round": round_id,
        "agreement_pct": agreement_pct,
        "disagreement_pct": disagreement_pct,
        "dominant_stance": dominant,
        "moderator": mod_payload,
    }

    round_summaries.append(round_summary_entry)

    print(
        f"[ROUND SUMMARY] Round {round_id}: "
        f"agreement={agreement_pct}% | disagreement={disagreement_pct}% | dominant={dominant}"
    )

# Final summary
final_input = {
    "topic": TOPIC,
    "all_round_summaries": round_summaries,
    "full_conversation": conversation_log,
}

FINAL_MODERATOR_PROMPT = """
You are the moderator of a multi-agent discussion.

Create a final summary of the whole discussion.

Your job:
- do NOT take a side
- summarize the overall evolution of the discussion
- explain whether the group moved toward consensus or stayed divided
- identify the strongest points raised
- keep it concise and readable

Output format:
Return valid JSON only with this structure:
{
  "final_summary": "overall summary",
  "consensus_assessment": "brief statement",
  "strongest_points": ["point 1", "point 2", "point 3"]
}
"""

print("[RUN] Generating final moderator summary...")
final_mod_raw = run_json_agent(
    FINAL_MODERATOR_PROMPT,
    f"Here is the full discussion data:\n{json.dumps(final_input, ensure_ascii=False, indent=2)}"
)
final_summary = {
    "final_summary": str(final_mod_raw.get("final_summary", "")).strip(),
    "consensus_assessment": str(final_mod_raw.get("consensus_assessment", "")).strip(),
    "strongest_points": final_mod_raw.get("strongest_points", []),
}
print("[OK] Final moderator summary generated.")

# Save
base_name = f"{timestamp}_{MODEL_NAME}_3agents_moderator"

result = {
    "timestamp": timestamp,
    "model": MODEL_NAME,
    "temperature": TEMPERATURE,
    "rounds": ROUNDS,
    "topic": TOPIC,
    "agents": {
        "Agent A": AGENT_A_PROMPT,
        "Agent B": AGENT_B_PROMPT,
        "Agent C": AGENT_C_PROMPT,
    },
    "conversation_log": conversation_log,
    "round_summaries": round_summaries,
    "final_summary": final_summary,
}

json_path = LOG_DIR / f"{base_name}.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

txt_path = LOG_DIR / f"{base_name}.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Temperature: {TEMPERATURE}\n")
    f.write(f"Rounds: {ROUNDS}\n")
    f.write(f"Topic: {TOPIC}\n")

    f.write("\n--- Conversation ---\n")
    for item in conversation_log:
        f.write(
            f'\nRound {item["round"]} | {item["speaker"]} '
            f'| stance={item["stance"]} | confidence={item["confidence"]}\n'
        )
        f.write(f'{item["message"]}\n')

    f.write("\n--- Moderator Round Summaries ---\n")
    for item in round_summaries:
        f.write(
            f'\nRound {item["round"]}: '
            f'agreement={item["agreement_pct"]}% | '
            f'disagreement={item["disagreement_pct"]}% | '
            f'dominant={item["dominant_stance"]}\n'
        )
        f.write(f'Summary: {item["moderator"]["round_summary"]}\n')
        f.write(f'Agreements: {item["moderator"]["main_agreements"]}\n')
        f.write(f'Disagreements: {item["moderator"]["main_disagreements"]}\n')
        f.write(f'Dominant theme: {item["moderator"]["dominant_theme"]}\n')

    f.write("\n--- Final Summary ---\n")
    f.write(f'{final_summary["final_summary"]}\n')
    f.write(f'Consensus assessment: {final_summary["consensus_assessment"]}\n')
    f.write(f'Strongest points: {final_summary["strongest_points"]}\n')

print("[DONE] Experiment finished.")
print(f"[SAVED] JSON log: {json_path}")
print(f"[SAVED] TXT log:  {txt_path}")