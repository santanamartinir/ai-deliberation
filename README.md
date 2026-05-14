# Multi-Agent Deliberation Framework

A small, local multi-agent deliberation framework built with **Ollama** and **LangChain**.

This project simulates discussions between multiple LLM-based agents with different identities and positions. It supports several discussion protocols, a moderator node, automatic logging, and basic post-hoc evaluation.

## Features

- 3 identity-conditioned agents
- 1 moderator node
- 3 discussion protocols:
  - `direct`
  - `moderated`
  - `delphi`
- 3 communication topologies:
  - `fully_connected`
  - `sequential`
  - `star`
- Automatic logging to `JSON` and `TXT`
- Basic automatic evaluation:
  - agreement / disagreement
  - confidence
  - stance shifts
  - lexical diversity
  - repeated openings

## Project Structure

```text
project/
│
├── main.py
├── prompts.py
├── protocols.py
├── metrics.py
├── utils.py
├── evaluation.py
└── logs/
```

## Requirements

- Python 3.10+
- **Ollama** installed and running locally
- A local model downloaded in **Ollama**, in this case:
    - `mistral`
    - `gemma`

## Installation

1. (optional) Create and activate a virtual environment

Windows PowerShell
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Make sure Ollama is running

```
ollama list
```

## How to Run

### Command-Line Arguments

`main.py` supports the following arguments:
- `--model`
    - local **Ollama** model name
    - `mistral`, `gemma`
- `--topic`
    - discussion topic
    - default: `Should universities integrate AI as a core learning tool?`
- `--rounds`
    - number of discussion rounds
    - default: `3`
- `--temperature`
    - sampling temperature for the model
    - default: `0.8`
- `--protocol`
    - discussion mode
    - options:
        - `direct`
        - `moderated`
        - `delphi`
- `--topology`
    - communication structure
    - options:
        - `fully_connected`
        - `sequential`
        - `star`

## Protocols

1. Direct

Agents respond directly to each other based on the visible discussion context.

2. Moderated

Agents discuss the topic and a moderator summarizes each round.

3. Delphi

Agents do not directly see the full raw debate. Instead, they mainly receive the moderator's anonymous summary from the previous round.

## Topologies

1. Fully Connected

All agents can see the recent conversation history.

2. Sequential

Agents mainly see messages produced earlier in the same round.

3. Star

Agents can see recent conversation history and the moderator summary.

## Evaluation

...

## Author

Irene Santana Martin

MSc Artificial Intelligence

University of Technology Nuremberg (UTN)

## License

GPL-3.0
