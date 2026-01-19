"""
Compliance Checker Module

Combines document analysis with image analysis to perform comprehensive
compliance checking against policy documents.

Example use cases:
- "Does this photo comply with workplace safety policy section 4.2?"
- "Check this building permit application against zoning regulations"
- "Verify this equipment setup meets OSHA requirements"
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retrieval import Citation, retrieve_relevant_chunks
from app.rag.multimodal_retrieval import ImageCitation, retrieve_multimodal, MultimodalRetriever
from app.rag.llms import get_llm, get_streaming_llm

settings = get_settings()
logger = get_logger(__name__)


class ComplianceStatus(str, Enum):
    """Compliance assessment status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NEEDS_REVIEW = "needs_review"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class ComplianceFinding:
    """Individual compliance finding."""
    id: str
    category: str
    status: ComplianceStatus
    description: str
    severity: str = "medium"  # low, medium, high, critical
    policy_reference: Optional[str] = None
    image_reference: Optional[str] = None
    recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "status": self.status.value,
            "description": self.description,
            "severity": self.severity,
            "policy_reference": self.policy_reference,
            "image_reference": self.image_reference,
            "recommendation": self.recommendation
        }


@dataclass
class ComplianceReport:
    """Complete compliance assessment report."""
    id: str
    title: str
    created_at: datetime
    overall_status: ComplianceStatus
    summary: str
    findings: List[ComplianceFinding]
    document_citations: List[Citation]
    image_citations: List[ImageCitation]
    query: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "overall_status": self.overall_status.value,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
            "document_citations": [c.to_dict() for c in self.document_citations],
            "image_citations": [c.to_dict() for c in self.image_citations],
            "query": self.query,
            "statistics": {
                "total_findings": len(self.findings),
                "compliant_count": sum(1 for f in self.findings if f.status == ComplianceStatus.COMPLIANT),
                "non_compliant_count": sum(1 for f in self.findings if f.status == ComplianceStatus.NON_COMPLIANT),
                "partial_count": sum(1 for f in self.findings if f.status == ComplianceStatus.PARTIAL),
                "needs_review_count": sum(1 for f in self.findings if f.status == ComplianceStatus.NEEDS_REVIEW),
                "documents_referenced": len(self.document_citations),
                "images_analyzed": len(self.image_citations)
            },
            "metadata": self.metadata
        }
    
    def to_markdown(self) -> str:
        """Generate a markdown-formatted compliance report."""
        lines = [
            f"# Compliance Report: {self.title}",
            f"**Generated:** {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Overall Status:** {self.overall_status.value.replace('_', ' ').title()}",
            "",
            "## Summary",
            self.summary,
            "",
            "## Findings",
        ]
        
        for i, finding in enumerate(self.findings, 1):
            status_emoji = {
                ComplianceStatus.COMPLIANT: "✅",
                ComplianceStatus.NON_COMPLIANT: "❌",
                ComplianceStatus.PARTIAL: "⚠️",
                ComplianceStatus.NEEDS_REVIEW: "🔍",
                ComplianceStatus.INSUFFICIENT_DATA: "❓"
            }.get(finding.status, "•")
            
            lines.append(f"\n### {i}. {finding.category} {status_emoji}")
            lines.append(f"**Status:** {finding.status.value.replace('_', ' ').title()}")
            lines.append(f"**Severity:** {finding.severity.title()}")
            lines.append(f"\n{finding.description}")
            
            if finding.policy_reference:
                lines.append(f"\n**Policy Reference:** {finding.policy_reference}")
            if finding.image_reference:
                lines.append(f"\n**Image Reference:** {finding.image_reference}")
            if finding.recommendation:
                lines.append(f"\n**Recommendation:** {finding.recommendation}")
        
        lines.append("\n## Sources")
        
        if self.document_citations:
            lines.append("\n### Documents Referenced")
            for doc in self.document_citations:
                lines.append(f"- {doc.filename} (relevance: {doc.score:.2%})")
        
        if self.image_citations:
            lines.append("\n### Images Analyzed")
            for img in self.image_citations:
                lines.append(f"- {img.filename} (relevance: {img.score:.2%})")
        
        return "\n".join(lines)


COMPLIANCE_SYSTEM_PROMPT = """You are a compliance analyst AI assistant that helps assess whether visual evidence (images, photos, diagrams) comply with policy documents and regulations.

Your task is to:
1. Analyze the provided policy document context
2. Consider the image descriptions provided
3. Assess compliance based on the user's specific question
4. Provide structured findings with clear status, references, and recommendations

Always be specific about which policy sections are relevant and why certain items are compliant or non-compliant.

Respond in the following JSON format:
{
    "overall_status": "compliant|non_compliant|partial|needs_review|insufficient_data",
    "summary": "Brief overall assessment summary",
    "findings": [
        {
            "category": "Category name (e.g., 'Safety Equipment', 'Fire Exit Access')",
            "status": "compliant|non_compliant|partial|needs_review|insufficient_data",
            "severity": "low|medium|high|critical",
            "description": "Detailed description of the finding",
            "policy_reference": "Specific policy section referenced (if applicable)",
            "image_reference": "Which image this finding relates to (if applicable)",
            "recommendation": "Specific action to take (for non-compliant items)"
        }
    ]
}

Be thorough but concise. If there's insufficient information to make a determination, indicate that clearly."""


class ComplianceChecker:
    """
    Multimodal compliance checker that combines document and image analysis.
    """
    
    def __init__(self, default_provider: str = "openai"):
        self.retriever = MultimodalRetriever(
            default_top_k_text=7,  # More context for compliance
            default_top_k_images=5
        )
        self.default_provider = default_provider
    
    def check_compliance(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        image_ids: Optional[List[str]] = None,
        provider: str = None,
        model: str = None,
        include_images_in_search: bool = True
    ) -> ComplianceReport:
        """
        Perform a compliance check combining documents and images.
        
        Args:
            query: The compliance question (e.g., "Does this photo comply with safety policy?")
            doc_ids: Specific documents to check against
            image_ids: Specific images to analyze
            provider: LLM provider to use
            model: Specific model name
            include_images_in_search: Whether to search for related images
        
        Returns:
            ComplianceReport with findings
        """
        import uuid
        
        provider = provider or self.default_provider
        report_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Starting compliance check: {query[:100]}...")
        
        # Step 1: Retrieve relevant documents and images
        retrieval_result = self.retriever.retrieve(
            query=query,
            include_images=include_images_in_search,
            doc_ids=doc_ids,
            image_ids=image_ids,
            top_k_text=7,
            top_k_images=5,
            use_hybrid=True  # Better coverage for compliance
        )
        
        text_citations = retrieval_result["text_citations"]
        image_citations = retrieval_result["image_citations"]
        context = retrieval_result["context"]
        
        # Step 2: Build the compliance analysis prompt
        analysis_prompt = self._build_analysis_prompt(query, context, image_citations)
        
        # Step 3: Get LLM analysis
        llm = get_llm(provider, model)
        messages = [
            {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
            {"role": "user", "content": analysis_prompt}
        ]
        
        try:
            response = llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the JSON response
            findings_data = self._parse_llm_response(response_text)
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            findings_data = {
                "overall_status": "insufficient_data",
                "summary": f"Analysis could not be completed: {str(e)}",
                "findings": []
            }
        
        # Step 4: Build the report
        findings = []
        for i, f in enumerate(findings_data.get("findings", [])):
            finding = ComplianceFinding(
                id=f"F{report_id}-{i+1}",
                category=f.get("category", "General"),
                status=ComplianceStatus(f.get("status", "needs_review")),
                description=f.get("description", ""),
                severity=f.get("severity", "medium"),
                policy_reference=f.get("policy_reference"),
                image_reference=f.get("image_reference"),
                recommendation=f.get("recommendation")
            )
            findings.append(finding)
        
        report = ComplianceReport(
            id=f"CR-{report_id}",
            title=f"Compliance Check: {query[:50]}...",
            created_at=datetime.utcnow(),
            overall_status=ComplianceStatus(findings_data.get("overall_status", "needs_review")),
            summary=findings_data.get("summary", ""),
            findings=findings,
            document_citations=text_citations,
            image_citations=image_citations,
            query=query,
            metadata={
                "provider": provider,
                "model": model,
                "doc_ids_filtered": doc_ids,
                "image_ids_filtered": image_ids
            }
        )
        
        logger.info(f"Compliance check complete: {report.overall_status.value}")
        return report
    
    def check_compliance_streaming(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        image_ids: Optional[List[str]] = None,
        provider: str = None,
        model: str = None,
        include_images_in_search: bool = True
    ):
        """
        Streaming version of compliance check.
        Yields progress events during analysis.
        """
        import uuid
        
        provider = provider or self.default_provider
        report_id = str(uuid.uuid4())[:8]
        
        # Yield: Starting
        yield {
            "type": "status",
            "data": {"stage": "retrieval", "message": "Retrieving relevant documents and images..."}
        }
        
        # Step 1: Retrieve
        retrieval_result = self.retriever.retrieve(
            query=query,
            include_images=include_images_in_search,
            doc_ids=doc_ids,
            image_ids=image_ids,
            top_k_text=7,
            top_k_images=5,
            use_hybrid=True
        )
        
        text_citations = retrieval_result["text_citations"]
        image_citations = retrieval_result["image_citations"]
        context = retrieval_result["context"]
        
        # Yield: Citations found
        yield {
            "type": "citations",
            "data": {
                "document_citations": [c.to_dict() for c in text_citations],
                "image_citations": [c.to_dict() for c in image_citations]
            }
        }
        
        yield {
            "type": "status",
            "data": {"stage": "analysis", "message": "Analyzing compliance..."}
        }
        
        # Step 2: Build prompt
        analysis_prompt = self._build_analysis_prompt(query, context, image_citations)
        
        # Step 3: Stream LLM response
        llm = get_streaming_llm(provider, model)
        messages = [
            {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
            {"role": "user", "content": analysis_prompt}
        ]
        
        full_response = ""
        try:
            for token in llm.stream(messages):
                if hasattr(token, 'content'):
                    token_text = token.content
                else:
                    token_text = str(token)
                full_response += token_text
                yield {"type": "token", "data": token_text}
            
            # Parse and yield final report
            findings_data = self._parse_llm_response(full_response)
            
            # Build findings
            findings = []
            for i, f in enumerate(findings_data.get("findings", [])):
                finding = ComplianceFinding(
                    id=f"F{report_id}-{i+1}",
                    category=f.get("category", "General"),
                    status=ComplianceStatus(f.get("status", "needs_review")),
                    description=f.get("description", ""),
                    severity=f.get("severity", "medium"),
                    policy_reference=f.get("policy_reference"),
                    image_reference=f.get("image_reference"),
                    recommendation=f.get("recommendation")
                )
                findings.append(finding)
            
            report = ComplianceReport(
                id=f"CR-{report_id}",
                title=f"Compliance Check: {query[:50]}...",
                created_at=datetime.utcnow(),
                overall_status=ComplianceStatus(findings_data.get("overall_status", "needs_review")),
                summary=findings_data.get("summary", ""),
                findings=findings,
                document_citations=text_citations,
                image_citations=image_citations,
                query=query,
                metadata={"provider": provider, "model": model}
            )
            
            yield {"type": "report", "data": report.to_dict()}
            
        except Exception as e:
            logger.error(f"Streaming compliance check failed: {e}")
            yield {"type": "error", "data": str(e)}
    
    def _build_analysis_prompt(
        self,
        query: str,
        context: str,
        image_citations: List[ImageCitation]
    ) -> str:
        """Build the analysis prompt for the LLM."""
        prompt_parts = [
            "## Compliance Analysis Request",
            f"\n**Question:** {query}",
            "\n## Policy Document Context",
            context or "No specific policy documents found.",
        ]
        
        if image_citations:
            prompt_parts.append("\n## Images to Analyze")
            for img in image_citations:
                desc = img.description or "No description available"
                prompt_parts.append(f"\n### Image: {img.filename}")
                prompt_parts.append(f"Description: {desc}")
        else:
            prompt_parts.append("\n## Images")
            prompt_parts.append("No images provided for analysis.")
        
        prompt_parts.append("\n## Instructions")
        prompt_parts.append(
            "Based on the policy documents and image descriptions above, "
            "analyze the compliance status and provide your findings in the required JSON format."
        )
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM's JSON response."""
        try:
            # Try to extract JSON from the response
            # Sometimes LLMs wrap JSON in markdown code blocks
            import re
            
            # Look for JSON in code blocks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response_text
            
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Return a basic structure with the raw response as summary
            return {
                "overall_status": "needs_review",
                "summary": response_text[:500],
                "findings": []
            }


# Default instance
compliance_checker = ComplianceChecker()


def check_compliance(
    query: str,
    doc_ids: Optional[List[str]] = None,
    image_ids: Optional[List[str]] = None,
    provider: str = "openai",
    model: str = None
) -> ComplianceReport:
    """
    Convenience function for compliance checking.
    
    Args:
        query: Compliance question
        doc_ids: Optional document filter
        image_ids: Optional image filter  
        provider: LLM provider
        model: Model name
    
    Returns:
        ComplianceReport
    """
    return compliance_checker.check_compliance(
        query=query,
        doc_ids=doc_ids,
        image_ids=image_ids,
        provider=provider,
        model=model
    )
