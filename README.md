# Cenotium

Agentic browser automation framework for AI-driven web interaction.

## Overview

Cenotium enables AI agents to autonomously navigate, interact with, and automate web tasks. It provides a structured layer that transforms standard web pages into agent-readable formats, enabling seamless automation and decision-making.

## Features

- **Web Schema Transformer**: Extracts website HTML, images, and UI components using vision and grounding models
- **LLM Compiler**: Task decomposition and parallel execution using LangChain and LangGraph
- **Multi-Agent Network**: Extensible agent architecture with Perplexity search, Twilio calling, and browser automation
- **Security Layer**: EigenTrust-based reputation system with Fernet encryption and HMAC-SHA256 signing

## Project Structure

```
cenotium/
├── src/
│   └── cenotium/
│       ├── agents/
│       │   ├── base/                 # Shared ReAct agent components
│       │   │   ├── callback_handler.py
│       │   │   ├── models.py
│       │   │   └── react_agent.py
│       │   ├── browser/              # Browser automation agent
│       │   │   ├── grounding.py
│       │   │   ├── sandbox_agent.py
│       │   │   ├── streaming.py
│       │   │   └── providers/
│       │   │       ├── base.py
│       │   │       ├── llm.py
│       │   │       └── osatlas.py
│       │   ├── perplexity/           # Web search agent
│       │   │   └── tools.py
│       │   └── twilio/               # Phone automation agent
│       │       └── tools.py
│       ├── compiler/                 # LLM Compiler for task planning
│       │   ├── executor.py
│       │   ├── llm_compiler.py
│       │   ├── output_parser.py
│       │   └── task_fetching.py
│       ├── orchestration/            # Agent orchestration
│       ├── security/                 # Security and trust management
│       │   ├── message_broker.py
│       │   ├── protocol.py
│       │   ├── storage.py
│       │   └── trust_core.py
│       ├── graph/                    # Graph database integration
│       │   └── neptune.py
│       ├── config/                   # Configuration management
│       └── utils/                    # Utility functions
├── servers/                          # Flask server implementations
│   ├── agents.py
│   ├── base.py
│   ├── compiler.py
│   └── orchestrator.py
├── benchmarks/                       # Benchmark configurations
│   ├── configs/
│   ├── datasets/
│   └── runner.py
├── ui/                               # Streamlit dashboard
│   └── app.py
├── tests/                            # Test suite
├── assets/                           # Static assets
├── templates/                        # HTML templates
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- Redis (for persistent storage)

### Setup

```bash
# Clone the repository
git clone https://github.com/Shrey1306/cenotium.git
cd cenotium

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For browser agent
pip install -e ".[browser]"

# For UI
pip install -e ".[ui]"
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required for LLM providers
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Optional providers
PERPLEXITY_KEY=your-perplexity-key
GROQ_API_KEY=your-groq-key
FIREWORKS_API_KEY=your-fireworks-key

# For browser agent
E2B_API_KEY=your-e2b-key
HF_TOKEN=your-huggingface-token

# For Twilio agent
ACCOUNT_SID=your-twilio-sid
TWILIO_KEY=your-twilio-key
TWILIO_FROM_NUMBER=+18778515935

# For graph database
NEPTUNE_ENDPOINT=your-neptune-endpoint
```

## Usage

### Running the Compiler Server

```bash
python -m servers.compiler
# Server runs on http://localhost:5000
```

### Running the Orchestrator

```bash
python -m servers.orchestrator
# Server runs on http://localhost:8080
```

### Running the UI

```bash
streamlit run ui/app.py
```

### Using the LLM Compiler

```python
from src.cenotium.compiler import LLMCompiler

compiler = LLMCompiler()
result = compiler.run("Plan a trip to Cabo for 8 people, under $1500 per person")
print(result)
```

### Using Agents

```python
from src.cenotium.agents.perplexity import perplexity_tool
from src.cenotium.agents.twilio import twilio_tool

# Search the web
result = perplexity_tool.func("What are the best beaches in Cabo?")

# Make a call
twilio_tool.run({"to_number": "+14155551234", "message": "Your flight is booked!"})
```

## Technologies

- **Backend**: Python, Flask
- **AI/ML**: LangChain, LangGraph, OpenAI GPT-4, Anthropic Claude
- **Computer Vision**: OS-Atlas, ShowUI
- **Security**: Fernet encryption, HMAC-SHA256
- **Storage**: Redis, Supabase
- **Graph Database**: AWS Neptune

## Contributors

- [Abhishek Pillai](https://github.com/abhipi)
- [Shrey Gupta](https://github.com/Shrey1306)
- [Ayush Gharat](https://github.com/ayushgharat)
- [Aditya Bajoria](https://github.com/Bajo-Adi)

## License

MIT License
