"""
Supervisor Agent - orchestrates the multi-agent workflow.

Responsibilities:
- Analyze user requests
- Route tasks to appropriate agents
- Monitor workflow progress
- Make decisions on next steps
- Handle workflow completion
"""
from typing import Dict, Any, Literal
import time
import json

from .base_agent import BaseAgent
from config import AgentPrompts
from state.schemas import AgentState, SupervisorDecision


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that orchestrates the workflow.
    
    Decision-making process:
    1. Analyze current state and task
    2. Review previous agent outputs
    3. Determine next agent to invoke
    4. Provide clear instructions
    5. Decide when workflow is complete
    """
    
    def __init__(self):
        super().__init__(name="supervisor")
        self.decision_history = []
    
    # def execute(self, state: AgentState) -> Dict[str, Any]:
    #     """
    #     Execute supervisor logic to route workflow.
        
    #     Args:
    #         state: Current workflow state
            
    #     Returns:
    #         State updates including next_agent decision
    #     """
    #     start_time = time.time()
    #     self.log_execution_start(state)
        
    #     try:
    #         # Build context from state
    #         context = self._build_context(state)
            
    #         # Get decision from LLM
    #         decision = self._make_routing_decision(state, context)
            
    #         # Record decision
    #         self.decision_history.append({
    #             'iteration': state['iteration_count'],
    #             'decision': decision,
    #             'timestamp': time.time()
    #         })
            
    #         # Log decision
    #         self.logger.info(
    #             f"Routing decision: {decision['next_agent']} (confidence: {decision['confidence']:.2f})"
    #         )
            
    #         # Prepare state updates
    #         updates = {
    #             'current_agent': self.name,
    #             'next_agent': decision['next_agent']
    #         }
            
    #         # Add message about decision
    #         message_update = self.add_message_to_state(
    #             state,
    #             recipient=decision['next_agent'],
    #             content=decision['instructions'],
    #             message_type="task"
    #         )
    #         updates.update(message_update)
            
    #         duration_ms = (time.time() - start_time) * 1000
    #         self.log_execution_end(state, duration_ms)
            
    #         return updates
            
    #     except Exception as e:
    #         self.logger.error(f"Supervisor execution failed: {str(e)}")
    #         return {
    #             'current_agent': self.name,
    #             'next_agent': 'finish',
    #             'errors': [f"Supervisor error: {str(e)}"]
    #         }
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute supervisor logic to route workflow.
        
        Args:
            state: Current workflow state
            
        Returns:
            State updates including next_agent decision
        """
        start_time = time.time()
        self.log_execution_start(state)
        
        try:
            # Check if we should force finish
            should_finish = (
                state['iteration_count'] >= self.settings.max_iterations or
                state['revision_count'] >= 3 or
                (state['review_decision'] == 'approve' and state.get('final_output'))
            )
            
            if should_finish:
                self.logger.info("Forcing workflow to finish")
                
                # If we have content but no final_output, set it
                updates = {
                    'current_agent': self.name,
                    'next_agent': 'finish',
                    'workflow_status': 'completed'
                }
                
                if state.get('content_draft') and not state.get('final_output'):
                    updates['final_output'] = state['content_draft']
                    updates['metadata'] = {
                        'word_count': len(state['content_draft'].split()),
                        'quality_score': state.get('review_score', 0.0),
                        'sources_used': len(state.get('research_sources', [])),
                        'revisions': state['revision_count'],
                        'status': 'completed_with_max_iterations'
                    }
                
                return updates

            # Deterministic routing for required workflow stages
            if state['content_draft'] and state['fact_check_status'] != 'completed':
                decision = SupervisorDecision(
                    next_agent='fact_checker',
                    reasoning='Content created but fact checking is not completed',
                    instructions='Fact-check the content claims against reliable web evidence',
                    confidence=1.0
                )
            else:
                # Build context from state
                context = self._build_context(state)

                # Get decision from LLM
                decision = self._make_routing_decision(state, context)
            
            
            
            # Record decision
            self.decision_history.append({
                'iteration': state['iteration_count'],
                'decision': decision,
                'timestamp': time.time()
            })
            
            # Log decision
            self.logger.info(
                f"Routing decision: {decision['next_agent']} (confidence: {decision['confidence']:.2f})"
            )
            
            # Prepare state updates
            updates = {
                'current_agent': self.name,
                'next_agent': decision['next_agent']
            }
            
            # Add message about decision
            message_update = self.add_message_to_state(
                state,
                recipient=decision['next_agent'],
                content=decision['instructions'],
                message_type="task"
            )
            updates.update(message_update)
            
            duration_ms = (time.time() - start_time) * 1000
            self.log_execution_end(state, duration_ms)
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Supervisor execution failed: {str(e)}")
            return {
                'current_agent': self.name,
                'next_agent': 'finish',
                'errors': [f"Supervisor error: {str(e)}"]
            }


    def _build_context(self, state: AgentState) -> str:
        """
        Build context summary for decision-making.
        
        Args:
            state: Current state
            
        Returns:
            Formatted context string
        """
        context_parts = [
            f"=== WORKFLOW CONTEXT ===",
            f"Task: {state['task']}",
            f"Iteration: {state['iteration_count']}/{self.settings.max_iterations}",
            f"Status: {state['workflow_status']}",
            ""
        ]
        
        # Add research status
        if state['research_findings']:
            context_parts.append(f"✓ Research completed (quality: {state['research_quality_score']:.2f})")
            context_parts.append(f"  Sources: {len(state['research_sources'])}")
        else:
            context_parts.append("✗ Research not yet completed")
        
        # Add content status
        if state['content_draft']:
            context_parts.append(f"✓ Content draft created (version {state['content_version']})")
        else:
            context_parts.append("✗ Content not yet created")

                # Add fact-check status
        if state['fact_check_status'] == 'completed':
            context_parts.append(
                f"✓ Fact checking completed (score: {state['fact_check_score']:.2f})"
            )
        elif state['fact_check_status'] == 'failed':
            context_parts.append("✗ Fact checking failed")
        else:
            context_parts.append("✗ Fact checking not yet completed")
        
        
        # Add review status
        if state['review_feedback']:
            context_parts.append(f"✓ Review completed (score: {state['review_score']:.2f})")
            context_parts.append(f"  Decision: {state['review_decision']}")
            context_parts.append(f"  Revisions: {state['revision_count']}")
        else:
            context_parts.append("✗ Review not yet completed")
        
        # Add recent messages (last 3)
        if state['messages']:
            context_parts.append("\n=== RECENT ACTIVITY ===")
            for msg in state['messages'][-3:]:
                context_parts.append(f"{msg.sender} → {msg.recipient}: {msg.content[:100]}...")
        
        # Add errors/warnings
        if state['errors']:
            context_parts.append(f"\n⚠ Errors: {len(state['errors'])}")
        
        return "\n".join(context_parts)
    
    def _make_routing_decision(self, state: AgentState, context: str) -> SupervisorDecision:
        """
        Make routing decision based on current state.
        
        Args:
            state: Current state
            context: Context summary
            
        Returns:
            SupervisorDecision with next agent and instructions
        """
        # Build prompt
        system_prompt = AgentPrompts.SUPERVISOR
        user_prompt = f"""
                    {context}

                    Based on the current state, decide the next step in the workflow.

                    You must respond with a JSON object in this exact format:
                    {{
                        "next_agent": "researcher|content_creator|fact_checker|reviewer|human_review|finish",
                        "reasoning": "explanation of why this decision was made",
                        "instructions": "specific instructions for the next agent",
                        "confidence": 0.0-1.0
                    }}

                    Decision rules:
                    - If no research done yet → "researcher"
                    - If research done but no content → "content_creator"
                    - If content created but fact checking is not completed → "fact_checker"
                    - If fact checking is completed but content has not been reviewed → "reviewer"
                    - If review failed and revisions < 3 → "content_creator" (with revision instructions)
                    - If review passed or revisions >= 3 → "finish"
                    - If critical error or max iterations → "finish"

                    Provide your decision:
                    """
        
        # Invoke LLM
        try:
            response = self.invoke_llm(system_prompt, user_prompt)
            
            # Parse JSON response
            # Clean markdown code blocks if present
            response = response.replace("```json", "").replace("```", "").strip()
            decision_data = json.loads(response)
            
            # Validate and create decision
            decision = SupervisorDecision(
                next_agent=decision_data.get('next_agent', 'finish'),
                reasoning=decision_data.get('reasoning', 'No reasoning provided'),
                instructions=decision_data.get('instructions', 'Proceed with task'),
                confidence=float(decision_data.get('confidence', 0.5))
            )
            
            return decision
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse supervisor decision: {e}")
            # Fallback decision based on state
            return self._fallback_decision(state)
        except Exception as e:
            self.logger.error(f"Error in routing decision: {e}")
            return self._fallback_decision(state)
    
    def _fallback_decision(self, state: AgentState) -> SupervisorDecision:
        """
        Fallback decision logic when LLM fails.
        
        Simple rule-based routing.
        """
        # Check iteration limit
        if state['iteration_count'] >= self.settings.max_iterations:
            self.logger.warning(f"Max iterations ({self.settings.max_iterations}) reached, finishing workflow")
            return SupervisorDecision(
                next_agent='finish',
                reasoning='Maximum iterations reached',
                instructions='Complete workflow',
                confidence=1.0
            )

        # Check if we've done too many revisions
        if state['revision_count'] >= 3:
            self.logger.warning(f"Max revisions (3) reached, finishing workflow")
            return SupervisorDecision(
                next_agent='finish',
                reasoning='Maximum revisions reached, accepting current content',
                instructions='Finalize output with current content',
                confidence=0.8
            )
            
        # Simple sequential logic
        if not state['research_findings']:
            return SupervisorDecision(
                next_agent='researcher',
                reasoning='No research completed yet',
                instructions=f"Research the topic: {state['task']}",
                confidence=0.9
            )
        
        if not state['content_draft']:
            return SupervisorDecision(
                next_agent='content_creator',
                reasoning='Research complete, need content',
                instructions='Create content based on research findings',
                confidence=0.9
            )

        if state['fact_check_status'] != 'completed':
            return SupervisorDecision(
                next_agent='fact_checker',
                reasoning='Content created but fact checking is not completed',
                instructions='Fact-check the content claims against reliable web evidence',
                confidence=0.9
            )

        if not state['review_feedback']:
            return SupervisorDecision(
                next_agent='reviewer',
                reasoning='Content created, needs review',
                instructions='Review content quality and accuracy',
                confidence=0.9
            )
        
        if state['review_decision'] == 'request_revision' and state['revision_count'] < 3:
            return SupervisorDecision(
                next_agent='content_creator',
                reasoning='Revision requested by reviewer',
                instructions=f"Revise content based on feedback: {state['review_feedback'][:200]}",
                confidence=0.8
            )
       # If review passed OR max revisions reached, finish
        return SupervisorDecision(
            next_agent='finish',
            reasoning='Workflow complete or max revisions reached',
            instructions='Finalize output',
            confidence=0.7
        )