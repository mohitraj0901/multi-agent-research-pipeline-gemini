"""
Content Creator Agent - generates high-quality content.

Responsibilities:
- Transform research into engaging content
- Adapt tone and style to audience
- Structure content effectively
- Incorporate citations properly
- Handle revision requests
"""
from typing import Dict, Any
import time

from .base_agent import BaseAgent
from config import AgentPrompts
from state.schemas import AgentState, ContentOutput


class ContentCreatorAgent(BaseAgent):
    """
    Content creator agent that generates written content.
    
    Process:
    1. Review research findings
    2. Plan content structure
    3. Generate content with proper citations
    4. Format for readability
    5. Handle revisions if needed
    """
    
    def __init__(self):
        super().__init__(name="content_creator")
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute content creation process.
        
        Args:
            state: Current workflow state
            
        Returns:
            State updates with content draft
        """
        start_time = time.time()
        self.log_execution_start(state)
        
        try:
            # Get research and any revision feedback
            research = state['research_findings']
            content_type = state.get('content_type', 'blog post')
            revision_feedback = state.get('review_feedback', '')
            is_revision = state['content_version'] > 0
            
            self.logger.info(
                f"Creating content (version {state['content_version'] + 1}, "
                f"type: {content_type}, revision: {is_revision})"
            )
            
            # Generate content
            content_output = self._generate_content(
                research=research,
                content_type=content_type,
                revision_feedback=revision_feedback,
                is_revision=is_revision
            )
            
            # Prepare state updates
            updates = {
                'current_agent': self.name,
                'content_draft': content_output['content'],
                'content_version': state['content_version'] + 1,
                'content_type': content_type,
                'fact_check_status': 'pending',
            }
            
            # Add completion message
            message_update = self.add_message_to_state(
                state,
                recipient='supervisor',
                content=f"Content created ({content_output['word_count']} words, "
                       f"{content_output['sources_cited']} sources cited)",
                message_type="result"
            )
            updates.update(message_update)
            
            duration_ms = (time.time() - start_time) * 1000
            self.log_execution_end(state, duration_ms)
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Content creation failed: {str(e)}")
            return {
                'current_agent': self.name,
                'content_draft': f"Content creation failed: {str(e)}",
                'errors': [f"Content creation error: {str(e)}"]
            }
    
    def _generate_content(
        self,
        research: str,
        content_type: str,
        revision_feedback: str = "",
        is_revision: bool = False
    ) -> ContentOutput:
        """
        Generate content based on research.
        
        Args:
            research: Research findings
            content_type: Type of content to create
            revision_feedback: Feedback for revisions
            is_revision: Whether this is a revision
            
        Returns:
            ContentOutput with generated content
        """
        # Build prompt
        system_prompt = AgentPrompts.CONTENT_CREATOR
        
        if is_revision:
            user_prompt = f"""
You are revising previously created content based on reviewer feedback.

Original Research:
{research}

Reviewer Feedback:
{revision_feedback}

Please revise the content addressing all feedback points while maintaining quality and accuracy.
Ensure all citations are properly included.

Create the revised {content_type}:
"""
        else:
            user_prompt = f"""
Create a high-quality {content_type} based on the research findings below.

Research Findings:
{research}

Requirements:
1. Engaging introduction that hooks the reader
2. Well-structured body with clear sections
3. Support claims with evidence from research
4. Include proper citations [1], [2], etc.
5. Compelling conclusion
6. Professional tone appropriate for general audience

Create the {content_type}:
"""
        
        try:
            # Generate content
            content = self.invoke_llm(system_prompt, user_prompt)
            
            # Analyze content
            word_count = len(content.split())
            sections = self._extract_sections(content)
            sources_cited = self._count_citations(content)
            
            self.logger.info(
                f"Content generated: {word_count} words, "
                f"{len(sections)} sections, {sources_cited} citations"
            )
            
            return ContentOutput(
                content=content,
                word_count=word_count,
                sections=sections,
                sources_cited=sources_cited
            )
            
        except Exception as e:
            self.logger.error(f"Content generation failed: {str(e)}")
            raise
    
    def _extract_sections(self, content: str) -> list[str]:
        """Extract section headers from content."""
        import re
        
        # Look for markdown headers or numbered sections
        headers = re.findall(r'^#+\s+(.+)$|^(\d+\.\s+.+)$', content, re.MULTILINE)
        sections = [h[0] or h[1] for h in headers if h[0] or h[1]]
        
        return sections[:10]  # Max 10 sections
    
    def _count_citations(self, content: str) -> int:
        """Count citation references in content."""
        import re
        
        # Count [1], [2], etc. style citations
        citations = re.findall(r'\[\d+\]', content)
        return len(set(citations))  # Unique citations