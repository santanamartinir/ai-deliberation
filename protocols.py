"""
protocols.py

This file contains the logic for the different discussion protocols:
- direct
- moderated
- delphi

Each protocol defines how agents see previous information and how the
moderator is used.
"""

import json

from langchain_core.messages import SystemMessage, HumanMessage

from prompts import AGENT_PROMPTS, FALLBACK_STANCES, MODERATOR_PROMPT
from metrics import compute_agreement, dominant_stance
from utils import parse_json_response, clean_payload, build_recent_context


def run_json_call(llm, system_prompt, user_prompt):
    """
    Send one request to the model and return parsed JSON.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return parse_json_response(response.content)


def get_visible_messages_for_agent(agent_name, round_outputs, conversation_log, topology, moderator_summary):
    """
    Decide what one agent can see depending on the topology.

    Topology options:
    - fully_connected: everyone sees recent conversation
    - sequential: an agent mainly sees earlier speakers in the same round
    - star: recent conversation + moderator summary
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