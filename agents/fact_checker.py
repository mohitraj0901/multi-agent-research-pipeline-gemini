"""
Fact Checker Agent.

Validates important claims in generated content against
web sources using Tavily and Gemini.
"""

from typing import Dict, Any, List
import re
import time

from .base_agent import BaseAgent
from config import AgentPrompts
from state.schemas import AgentState
from tools.web_search import web_search_tool


class FactCheckerAgent(BaseAgent):
    """
    Fact-checks claims from the generated content.

    Process:
    1. Extract important factual claims
    2. Search the web for supporting evidence
    3. Ask Gemini to evaluate the claims
    4. Produce a fact-check report
    """

    def __init__(self):
        super().__init__(name="fact_checker")

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """Execute fact-checking process."""

        start_time = time.time()
        self.log_execution_start(state)

        try:
            content = state.get("content_draft", "")

            if not content:
                return {
                    "current_agent": self.name,
                    "fact_check_report": "No content available for fact checking.",
                    "fact_check_score": 0.0,
                    "fact_check_status": "failed",
                    "errors": ["No content draft available"]
                }

            # Extract claims from generated content
            claims = self._extract_claims(content)

            # Search for supporting evidence
            evidence = self._collect_evidence(claims)

            # Evaluate claims using Gemini
            report = self._evaluate_claims(content, claims, evidence)

            # Calculate a simple factuality score
            score = self._calculate_score(report)

            updates = {
                "current_agent": self.name,
                "fact_check_report": report,
                "fact_check_score": score,
                "fact_check_status": "completed",
            }

            message_update = self.add_message_to_state(
                state,
                recipient="supervisor",
                content=f"Fact checking completed. Score: {score:.2f}",
                message_type="result"
            )

            updates.update(message_update)

            duration_ms = (time.time() - start_time) * 1000
            self.log_execution_end(state, duration_ms)

            return updates

        except Exception as e:
            self.logger.error(f"Fact checking failed: {str(e)}")

            return {
                "current_agent": self.name,
                "fact_check_report": f"Fact checking failed: {str(e)}",
                "fact_check_score": 0.0,
                "fact_check_status": "failed",
                "errors": [f"Fact checker error: {str(e)}"]
            }

    def _extract_claims(self, content: str, max_claims: int = 8) -> List[str]:
        """Extract potentially verifiable factual claims."""

        sentences = re.split(r"(?<=[.!?])\s+", content)

        claims = []

        for sentence in sentences:
            sentence = sentence.strip()

            if len(sentence) < 40:
                continue

            # Focus on sentences containing factual indicators
            factual_indicators = [
                "%",
                "2017",
                "2020",
                "2021",
                "2022",
                "2023",
                "2024",
                "2025",
                "2026",
                "according to",
                "research",
                "study",
                "paper",
                "introduced",
                "developed",
                "increased",
                "decreased",
                "first",
                "largest",
                "most",
                "only"
            ]

            if any(
                indicator.lower() in sentence.lower()
                for indicator in factual_indicators
            ):
                claims.append(sentence)

            if len(claims) >= max_claims:
                break

        # If no indicator-based claims were found,
        # use the first few substantive sentences.
        if not claims:
            claims = [
                s.strip()
                for s in sentences
                if len(s.strip()) >= 60
            ][:max_claims]

        return claims

    def _collect_evidence(self, claims: List[str]) -> str:
        """Search Tavily for evidence supporting or challenging claims."""

        evidence_parts = []

        for i, claim in enumerate(claims, 1):
            try:
                # Keep search query reasonably short
                query = claim[:300]

                results = web_search_tool(
                    query,
                    num_results=3
                )

                evidence_parts.append(
                    f"\nClaim {i}:\n{claim}\n"
                    f"Web Evidence:\n{results[:1500]}"
                )

            except Exception as e:
                evidence_parts.append(
                    f"\nClaim {i}:\n{claim}\n"
                    f"Evidence search failed: {str(e)}"
                )

        return "\n".join(evidence_parts)

    def _evaluate_claims(
        self,
        content: str,
        claims: List[str],
        evidence: str
    ) -> str:
        """Use Gemini to evaluate claims against web evidence."""

        system_prompt = """
You are a rigorous fact-checking assistant.

Evaluate factual claims in the provided content against the
web evidence supplied.

For each claim:
1. Label it as SUPPORTED, PARTIALLY SUPPORTED, or UNSUPPORTED.
2. Briefly explain why.
3. Identify the supporting evidence when available.
4. Flag claims that need correction or stronger sourcing.

Do not invent evidence.
If the evidence is insufficient, explicitly say so.

Finish with:
- Overall assessment
- Claims requiring revision
"""

        user_prompt = f"""
CONTENT:
{content[:6000]}

CLAIMS TO VERIFY:
{chr(10).join(
    f"{i}. {claim}"
    for i, claim in enumerate(claims, 1)
)}

WEB EVIDENCE:
{evidence[:10000]}
"""

        return self.invoke_llm(
            system_prompt,
            user_prompt
        )

    def _calculate_score(self, report: str) -> float:
        """Calculate a simple fact-check score from the report."""

        if not report:
            return 0.0

        partial = len(
            re.findall(
                r"\bPARTIALLY SUPPORTED\b",
                report,
                re.IGNORECASE
            )
        )

        unsupported = len(
            re.findall(
                r"\bUNSUPPORTED\b",
                report,
                re.IGNORECASE
            )
        )

        supported = len(
            re.findall(
                r"(?<!PARTIALLY )\bSUPPORTED\b",
                report,
                re.IGNORECASE
            )
        )

        total = supported + partial + unsupported

        if total == 0:
            return 0.5

        return (supported + 0.5 * partial) / total