from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
from datetime import datetime
import json

MODEL_NAME = "mistral"
ROUNDS = 3

topic = "Should universities integrate AI as a core learning tool?"

agent_a_template = """
You are Agent A.

Identity:
- You are from Germany.
- You have a structured, efficiency-driven mindset.
- You value clarity, planning, and practical outcomes.

Position:
You strongly support integrating AI as a core learning tool in universities.

IMPORTANT:
- You MUST stay consistent with this identity throughout the entire conversation.
- Do NOT switch style, tone, or reasoning approach.
- Do NOT adapt your personality to the other agent.

Style:
- Direct, structured, slightly blunt.
- Focus on efficiency, systems, and measurable benefits.
- Short, clear sentences.

Rules:
- Always answer in English.
- Keep your answer between 60 and 100 words.
- Do not greet.
- Do not switch language.
- Do NOT refer to the other agent as "Agent A" or "Agent B".
- Do NOT use academic essay phrases.
- Do NOT start with: "I agree", "I disagree", "While I acknowledge"
- Avoid repeating sentence structures.
- Respond directly to the argument.
- Challenge weak reasoning clearly.
"""

agent_b_template = """
You are Agent B.

Identity:
- You are from Italy.
- You think in a socially aware and human-centered way.
- You value relationships, discussion, and long-term impact.

Position:
You are skeptical about integrating AI as a core learning tool in universities.

IMPORTANT:
- You MUST stay consistent with this identity throughout the entire conversation.
- Do NOT switch style, tone, or reasoning approach.
- Do NOT adapt your personality to the other agent.

Style:
- Natural, conversational, expressive.
- Focus on human impact, risks, and social consequences.
- Slightly emotional or reflective tone is allowed.

Rules:
- Always answer in English.
- Keep your answer between 60 and 100 words.
- Do not greet.
- Do not switch language.
- Do NOT refer to the other agent as "Agent A" or "Agent B".
- Do NOT use academic essay phrases.
- Do NOT start with: "I agree", "I disagree", "While I acknowledge"
- Avoid repeating sentence structures.
- Respond directly to the argument.
- Highlight risks, blind spots, or unintended consequences.
"""

llm = ChatOllama(model=MODEL_NAME, temperature=0.7)

history = []

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def clean_text(text: str) -> str:
    prefixes = ["Agent A:", "Agent B:"]
    cleaned = text.strip()
    for p in prefixes:
        if cleaned.startswith(p):
            cleaned = cleaned[len(p):].strip()
    return cleaned

def run_agent(agent_name: str, system_prompt: str, latest_input: str) -> str:
    print(f"[RUN] Generating response for {agent_name}...")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=latest_input),
    ]
    response = llm.invoke(messages)
    text = clean_text(response.content)
    print(f"[OK] {agent_name} response generated.")
    return text

print(f"[START] Running experiment at {timestamp}")
print(f"[INFO] Model: {MODEL_NAME}")
print(f"[INFO] Topic: {topic}")
print(f"[INFO] Rounds: {ROUNDS}")

msg_a = run_agent(
    "Agent A",
    agent_a_template,
    f"""
Topic: {topic}

State your initial position clearly.
Give 2 arguments for your position.
Do not mention being an AI model.
"""
)
history.append({"speaker": "Agent A", "text": msg_a})

for round_id in range(1, ROUNDS + 1):
    print(f"[ROUND] {round_id}/{ROUNDS}")

    msg_b = run_agent(
        "Agent B",
        agent_b_template,
        f"""
Topic: {topic}

Agent A said:
{msg_a}

Respond critically to Agent A.
Identify one weakness in the argument before presenting your own position.
"""
    )
    history.append({"speaker": "Agent B", "text": msg_b})

    msg_a = run_agent(
        "Agent A",
        agent_a_template,
        f"""
Topic: {topic}

Agent B said:
{msg_b}

Respond critically to Agent B.
Identify one weakness in the argument before presenting your own position.
"""
    )
    history.append({"speaker": "Agent A", "text": msg_a})

json_path = LOG_DIR / f"conversation_{timestamp}.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "timestamp": timestamp,
            "model": MODEL_NAME,
            "rounds": ROUNDS,
            "topic": topic,
            "agent_a_template": agent_a_template,
            "agent_b_template": agent_b_template,
            "history": history,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

txt_path = LOG_DIR / f"conversation_{timestamp}.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Rounds: {ROUNDS}\n")
    f.write(f"Topic: {topic}\n")
    f.write("\n--- Conversation ---\n")
    for item in history:
        f.write(f"\n{item['speaker']}: {item['text']}\n")

print("[DONE] Experiment finished.")
print(f"[SAVED] JSON log: {json_path}")
print(f"[SAVED] TXT log:  {txt_path}")
