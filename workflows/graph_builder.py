"""
LangGraph workflow builder for multi-agent system.

Demonstrates:
- StateGraph construction
- Conditional edges for routing
- Agent node integration
- Checkpointing setup
- Workflow compilation
"""
from typing import Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state.schemas import AgentState, increment_iteration, update_workflow_status
from agents import SupervisorAgent, ResearcherAgent, ContentCreatorAgent, ReviewerAgent, FactCheckerAgent
from config import get_settings
from utils.logger import get_logger


class WorkflowBuilder:
    """
    Builds LangGraph workflow for multi-agent system.
    
    Workflow structure:
    - Start → Supervisor
    - Supervisor routes to: Researcher, Content Creator, Reviewer, or Finish
    - Each agent reports back to Supervisor
    - Supervisor makes next routing decision
    - Loop continues until completion
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("workflow_builder")
        
        # Initialize agents
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.content_creator = ContentCreatorAgent()
        self.reviewer = ReviewerAgent()
        self.fact_checker = FactCheckerAgent()
    
    def build(self, checkpointer: Optional[MemorySaver] = None) -> StateGraph:
        """
        Build the complete workflow graph.
        
        Args:
            checkpointer: Optional checkpointer for state persistence
            
        Returns:
            Compiled StateGraph
        """
        self.logger.info("Building multi-agent workflow graph")
        
        # Create graph with AgentState
        graph = StateGraph(AgentState)
        
        # Add agent nodes
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("researcher", self._researcher_node)
        graph.add_node("content_creator", self._content_creator_node)
        graph.add_node("fact_checker", self._fact_checker_node)
        graph.add_node("reviewer", self._reviewer_node)
        
        # Set entry point
        graph.set_entry_point("supervisor")
        
        # Add conditional edges from supervisor
        graph.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "researcher": "researcher",
                "content_creator": "content_creator",
                "fact_checker": "fact_checker",
                "reviewer": "reviewer",
                "finish": END
            }
        )
        
        # All agents route back to supervisor
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("content_creator", "supervisor")
        graph.add_edge("fact_checker", "supervisor")
        graph.add_edge("reviewer", "supervisor")
        
        # Compile graph
        compiled_graph = graph.compile(checkpointer=checkpointer)
        
        self.logger.info("Workflow graph compiled successfully")
        
        return compiled_graph
    
    def _supervisor_node(self, state: AgentState) -> dict:
        """
        Supervisor agent node.
        
        Executes supervisor logic and updates iteration count.
        """
        self.logger.debug("Executing supervisor node")
        
        # Check iteration limit
        if state['iteration_count'] >= self.settings.max_iterations:
            self.logger.warning(f"Max iterations ({self.settings.max_iterations}) reached")
            return {
                'next_agent': 'finish',
                'workflow_status': 'completed',
                'warnings': [f"Max iterations reached"]
            }
        
        # Execute supervisor
        updates = self.supervisor.execute(state)
        
        # Increment iteration
        iter_update = increment_iteration(state)
        updates.update(iter_update)
        
        return updates
    
    def _researcher_node(self, state: AgentState) -> dict:
        """Research agent node."""
        self.logger.debug("Executing researcher node")
        return self.researcher.execute(state)
    
    def _content_creator_node(self, state: AgentState) -> dict:
        """Content creator agent node."""
        self.logger.debug("Executing content creator node")
        return self.content_creator.execute(state)
    
    def _reviewer_node(self, state: AgentState) -> dict:
        """Reviewer agent node."""
        self.logger.debug("Executing reviewer node")
        return self.reviewer.execute(state)

    def _fact_checker_node(self, state: AgentState) -> dict:
        """Fact checker agent node."""
        self.logger.debug("Executing fact checker node")
        return self.fact_checker.execute(state)
    
    def _route_from_supervisor(
        self,
        state: AgentState
    ) -> Literal["researcher", "content_creator", "fact_checker", "reviewer", "finish"]:
        """
        Conditional routing logic from supervisor.
        
        Routes based on supervisor's next_agent decision.
        
        Args:
            state: Current state
            
        Returns:
            Next node name
        """
        next_agent = state.get('next_agent', 'finish')
        
        self.logger.info(f"Routing from supervisor to: {next_agent}")
        
        # Map to valid edges
        route_map = {
            'researcher': 'researcher',
            'content_creator': 'content_creator',
            'fact_checker': 'fact_checker',
            'reviewer': 'reviewer',
            'human_review': 'finish',  # Not implemented, end workflow
            'finish': 'finish'
        }
        
        return route_map.get(next_agent, 'finish')


def create_workflow(with_checkpointing: bool = True) -> StateGraph:
    """
    Create and return compiled workflow.
    
    Args:
        with_checkpointing: Enable state checkpointing
        
    Returns:
        Compiled StateGraph ready for execution
    """
    builder = WorkflowBuilder()
    
    checkpointer = None
    if with_checkpointing:
        from .checkpointer import get_checkpointer
        checkpointer = get_checkpointer()
    
    return builder.build(checkpointer=checkpointer)