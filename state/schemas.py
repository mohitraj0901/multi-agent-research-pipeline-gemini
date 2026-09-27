"""
State schemas using TypedDict for LangGraph state management.

Key Concepts Demonstrated:
- TypedDict for type-safe state management
- Annotation for state reducers
- Immutable state updates
- Structured message passing between agents
"""
from typing import TypedDict, Annotated, Sequence, Literal
from operator import add
from dataclasses import dataclass
from datetime import datetime


# Agent message for inter-agent communication
@dataclass
class AgentMessage:
    """
    Message passed between agents.
    
    Important Point:
    - Structured communication between agents
    - Tracks message flow for debugging
    - Immutable dataclass for safety
    """
    sender: str
    recipient: str
    content: str
    timestamp: datetime
    message_type: Literal["task", "result", "feedback", "question"]
    
    def __str__(self) -> str:
        return f"[{self.sender} → {self.recipient}] {self.content[:100]}..."


class AgentState(TypedDict):
    """
    Main state object passed through the LangGraph workflow.
    
    Important Points:
    - TypedDict provides type safety without runtime overhead
    - Annotated with 'add' operator for appending to lists (reducer pattern)
    - All agents read from and write to this shared state
    - Immutable updates ensure state consistency
    """
    
    # Input from user
    task: str  # Original user request
    user_context: dict  # Additional context from user
    
    # Workflow control
    current_agent: str  # Which agent is currently active
    next_agent: str  # Which agent should run next
    iteration_count: int  # Track number of iterations (prevent infinite loops)
    workflow_status: Literal["running", "waiting", "completed", "failed"]
    
    # Agent messages (using reducer pattern)
    messages: Annotated[Sequence[AgentMessage], add]
    
    # Research agent outputs
    research_query: str  # Query formulated by supervisor
    research_findings: str  # Raw research data
    research_sources: list[str]  # URLs and citations
    research_quality_score: float  # Self-assessment score
    
    # Content creator outputs
    content_type: str  # Type of content to create (blog, report, etc.)
    content_draft: str  # Generated content
    content_version: int  # Track revisions

    # Fact checker outputs
    fact_check_report: str
    fact_check_score: float
    fact_check_status: Literal["pending", "completed", "failed"]
    
    # Reviewer outputs
    review_score: float  # Overall quality score (0-10)
    review_feedback: str  # Detailed feedback
    review_decision: Literal["approve", "request_revision", "reject"]
    revision_count: int  # Track how many revisions requested
    
    # Final output
    final_output: str  # Final approved content
    metadata: dict  # Additional metadata (word count, sources used, etc.)
    
    # Error handling
    errors: Annotated[list[str], add]  # Collect any errors
    warnings: Annotated[list[str], add]  # Collect any warnings


class SupervisorDecision(TypedDict):
    """
    Supervisor agent decision output.
    
    Important Point:
    - Structured decision-making output
    - Clear routing logic
    - Supports conditional edges in LangGraph
    """
    next_agent: Literal["researcher", "content_creator", "fact_checker", "reviewer", "human_review", "finish"]
    reasoning: str  # Why this decision was made
    instructions: str  # Specific instructions for next agent
    confidence: float  # Confidence in decision (0-1)


class ResearchOutput(TypedDict):
    """Research agent structured output."""
    findings: str
    sources: list[str]
    quality_score: float
    knowledge_gaps: list[str]
    recommendations: str


class ContentOutput(TypedDict):
    """Content creator structured output."""
    content: str
    word_count: int
    sections: list[str]
    sources_cited: int


class ReviewOutput(TypedDict):
    """Reviewer structured output."""
    overall_score: float
    criterion_scores: dict[str, float]  # Individual criterion scores
    strengths: list[str]
    issues: list[str]
    recommendations: list[str]
    decision: Literal["approve", "request_revision", "reject"]


# Helper functions for state management

def create_initial_state(task: str, context: dict = None) -> AgentState:
    """
    Create initial state for workflow.
    
    Important Point:
    - Factory pattern for consistent state initialization
    - Provides sensible defaults
    - Ensures all required fields are present
    """
    return AgentState(
        task=task,
        user_context=context or {},
        current_agent="supervisor",
        next_agent="supervisor",
        iteration_count=0,
        workflow_status="running",
        messages=[],
        research_query="",
        research_findings="",
        research_sources=[],
        research_quality_score=0.0,
        content_type="blog post",
        content_draft="",
        content_version=0,
        fact_check_report="",
        fact_check_score=0.0,
        fact_check_status="pending",
        review_score=0.0,
        review_feedback="",
        review_decision="request_revision",
        revision_count=0,
        final_output="",
        metadata={},
        errors=[],
        warnings=[]
    )


def add_message(
    state: AgentState, 
    sender: str, 
    recipient: str, 
    content: str,
    message_type: Literal["task", "result", "feedback", "question"] = "task"
) -> dict:
    """
    Add a message to state (demonstrates reducer pattern).
    
    Important Point:
    - State updates return partial state (only changed fields)
    - Annotated fields with 'add' operator automatically append
    - Immutable update pattern (doesn't modify original state)
    """
    message = AgentMessage(
        sender=sender,
        recipient=recipient,
        content=content,
        timestamp=datetime.now(),
        message_type=message_type
    )
    return {"messages": [message]}


def increment_iteration(state: AgentState) -> dict:
    """Increment iteration count (safety check for infinite loops)."""
    return {"iteration_count": state["iteration_count"] + 1}


def update_workflow_status(
    status: Literal["running", "waiting", "completed", "failed"]
) -> dict:
    """Update workflow status."""
    return {"workflow_status": status}