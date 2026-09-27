# 🤖 Multi-Agent Research & Content Pipeline

A multi-agent AI research and content generation pipeline built with **LangGraph**, **Google Gemini**, and **Tavily**.

This project extends an existing multi-agent workflow with a Gemini-based LLM integration and an automated **Fact Checker agent** that verifies factual claims using web evidence before content reaches the final review stage.

## 🎯 Overview

The pipeline coordinates specialized agents through a shared state and Supervisor-based routing:

- **🎯 Supervisor Agent** — Orchestrates the workflow and decides which agent runs next
- **🔎 Researcher Agent** — Performs live web research using Tavily
- **✍️ Content Creator Agent** — Generates structured content from research findings
- **🔍 Fact Checker Agent** — Extracts factual claims, searches for supporting evidence, and evaluates claim support using Gemini
- **✅ Reviewer Agent** — Reviews content quality and provides revision feedback

### Workflow

```text
User Task
   │
   ▼
Supervisor
   │
   ▼
Researcher ──────► Tavily Web Search
   │
   ▼
Content Creator ─► Gemini
   │
   ▼
Fact Checker ────► Tavily Evidence + Gemini Evaluation
   │
   ▼
Reviewer
   │
   ├── Request Revision ──► Content Creator
   │                         │
   │                         ▼
   │                     Fact Checker
   │
   ▼
Final Output

🚀 Key Features
Multi-agent orchestration using LangGraph
Google Gemini integration using langchain-google-genai
Live web research using Tavily
Automated factual claim verification
Evidence-based fact-checking before final review
Iterative content revision through Supervisor routing
Shared typed workflow state using TypedDict/Pydantic-based schemas
Checkpointing and workflow state management
Error handling and execution logging
🔍 Fact Checker

The custom Fact Checker agent is responsible for an additional verification stage between content generation and final review.

Process
Extract important factual claims from the generated content
Search Tavily for supporting web evidence
Send the claims and retrieved evidence to Gemini
Classify claims as:
SUPPORTED
PARTIALLY SUPPORTED
UNSUPPORTED
Calculate a simple factuality score
Store the fact-check report and score in the shared workflow state

When content is revised, the fact-check status is reset so the updated draft can be checked again.

🧠 Technology Stack
Python
LangGraph
LangChain
Google Gemini
Tavily Search API
Pydantic
TypedDict
python-dotenv
📁 Project Structure
multi-agent-research-pipeline-gemini/
│
├── agents/
│   ├── base_agent.py
│   ├── researcher.py
│   ├── content_creator.py
│   ├── supervisor.py
│   └── fact_checker.py
│
├── config/
│   └── settings.py
│
├── state/
│   └── schemas.py
│
├── workflows/
│   └── graph_builder.py
│
├── tools/
├── utils/
├── tests/
├── examples/
├── screenshots/
│
├── main.py
├── requirements.txt
├── .env.example
└── README.md
⚙️ Quick Start
Prerequisites
Python 3.10+
Google Gemini API key
Tavily API key
Installation
git clone https://github.com/mohitraj0901/multi-agent-research-pipeline-gemini.git

cd multi-agent-research-pipeline-gemini

python -m venv .venv
Activate virtual environment

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Linux/macOS:

source .venv/bin/activate
Install dependencies
pip install -r requirements.txt
Configuration

Create a .env file in the project root:

GEMINI_API_KEY=your-gemini-api-key
TAVILY_API_KEY=your-tavily-api-key

LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4000

MAX_ITERATIONS=15

Never commit your .env file or API keys to GitHub.

Run
python main.py
🧪 Example Run

The pipeline can research a topic, generate content, fact-check the generated claims, and then send the content through the reviewer.

Example workflow:

RESEARCHER completed
        ↓
CONTENT_CREATOR completed
        ↓
FACT_CHECKER completed
        ↓
REVIEWER completed
        ↓
FINAL OUTPUT
📊 Validation

The modified pipeline was tested locally with a complete end-to-end workflow.

A successful run demonstrated:

Research through Tavily
Content generation through Gemini
Fact-checking with web evidence
Supervisor routing through the Fact Checker stage
Final reviewer approval
Successful final output generation

One local test run completed in 5 workflow iterations, with a reviewer quality score of 0.91/1.0.

The reported result is from a local test run and is not intended as a general benchmark.

📜 License

MIT License


