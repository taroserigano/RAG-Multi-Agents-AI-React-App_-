"""
Compliance checking API routes.
Provides endpoints for multimodal compliance assessment combining 
document analysis with image analysis.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from app.db.session import get_db
from app.db.models import ComplianceReport as ComplianceReportModel
from app.core.logging import get_logger
from app.rag.compliance_checker import (
    compliance_checker,
    ComplianceReport,
    ComplianceStatus
)

router = APIRouter(prefix="/api/compliance", tags=["compliance"])
logger = get_logger(__name__)


# ============================================================================
# Request/Response Schemas
# ============================================================================

class ComplianceCheckRequest(BaseModel):
    """Request schema for compliance check."""
    user_id: str = Field(..., description="User identifier")
    query: str = Field(..., min_length=1, description="Compliance question")
    provider: str = Field("openai", description="LLM provider to use")
    model: Optional[str] = Field(None, description="Specific model name")
    doc_ids: Optional[List[str]] = Field(None, description="Specific documents to check against")
    image_ids: Optional[List[str]] = Field(None, description="Specific images to analyze")
    include_image_search: bool = Field(True, description="Search for related images")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user-123",
                "query": "Does this workspace photo comply with our fire safety policy?",
                "provider": "openai",
                "doc_ids": ["doc-abc"],
                "image_ids": ["img-xyz"],
                "include_image_search": True
            }
        }


class ComplianceFindingResponse(BaseModel):
    """Individual compliance finding."""
    id: str
    category: str
    status: str
    description: str
    severity: str
    policy_reference: Optional[str] = None
    image_reference: Optional[str] = None
    recommendation: Optional[str] = None


class ComplianceReportResponse(BaseModel):
    """Response schema for compliance report."""
    id: str
    title: str
    created_at: datetime
    overall_status: str
    summary: str
    findings: List[ComplianceFindingResponse]
    document_citations: List[Dict[str, Any]]
    image_citations: List[Dict[str, Any]]
    query: str
    statistics: Dict[str, Any]


class ComplianceHistoryResponse(BaseModel):
    """Response for compliance check history."""
    id: str
    title: str
    created_at: datetime
    overall_status: str
    query: str
    finding_count: int


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/check", response_model=ComplianceReportResponse)
async def check_compliance(
    request: ComplianceCheckRequest,
    db: Session = Depends(get_db)
):
    """
    Perform a compliance check combining documents and images.
    
    This endpoint:
    1. Retrieves relevant policy documents based on the query
    2. Retrieves or uses specified images
    3. Analyzes compliance using an LLM
    4. Returns a detailed compliance report
    
    Example queries:
    - "Does this workspace photo comply with OSHA safety requirements?"
    - "Check this building permit against zoning regulations"
    - "Is this equipment installation compliant with section 4.2?"
    """
    logger.info(f"Compliance check request from user {request.user_id}")
    
    try:
        report = compliance_checker.check_compliance(
            query=request.query,
            doc_ids=request.doc_ids,
            image_ids=request.image_ids,
            provider=request.provider,
            model=request.model,
            include_images_in_search=request.include_image_search
        )
        
        # Save to database if available
        if db:
            try:
                db_report = ComplianceReportModel(
                    id=report.id,
                    user_id=request.user_id,
                    title=report.title,
                    query=report.query,
                    overall_status=report.overall_status.value,
                    summary=report.summary,
                    findings_json=json.dumps([f.to_dict() for f in report.findings]),
                    document_ids=request.doc_ids,
                    image_ids=request.image_ids,
                    provider=request.provider,
                    model=request.model
                )
                db.add(db_report)
                db.commit()
            except Exception as e:
                logger.warning(f"Failed to save compliance report to database: {e}")
        
        return report.to_dict()
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compliance check failed: {str(e)}"
        )


@router.post("/check/stream")
async def check_compliance_stream(
    request: ComplianceCheckRequest,
    db: Session = Depends(get_db)
):
    """
    Streaming compliance check with real-time progress updates.
    
    Returns Server-Sent Events (SSE) with:
    - status: Progress updates
    - citations: Retrieved sources
    - token: LLM response tokens
    - report: Final compliance report
    - error: Error messages
    """
    logger.info(f"Streaming compliance check from user {request.user_id}")
    
    def generate():
        try:
            for event in compliance_checker.check_compliance_streaming(
                query=request.query,
                doc_ids=request.doc_ids,
                image_ids=request.image_ids,
                provider=request.provider,
                model=request.model,
                include_images_in_search=request.include_image_search
            ):
                yield f"data: {json.dumps(event)}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming compliance check failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/history/{user_id}", response_model=List[ComplianceHistoryResponse])
async def get_compliance_history(
    user_id: str,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Get compliance check history for a user.
    """
    if not db:
        return []
    
    try:
        reports = db.query(ComplianceReportModel)\
            .filter(ComplianceReportModel.user_id == user_id)\
            .order_by(ComplianceReportModel.created_at.desc())\
            .limit(limit)\
            .all()
        
        return [
            {
                "id": r.id,
                "title": r.title,
                "created_at": r.created_at,
                "overall_status": r.overall_status,
                "query": r.query,
                "finding_count": len(json.loads(r.findings_json)) if r.findings_json else 0
            }
            for r in reports
        ]
    except Exception as e:
        logger.error(f"Failed to fetch compliance history: {e}")
        return []


@router.get("/report/{report_id}")
async def get_compliance_report(
    report_id: str,
    format: str = "json",
    db: Session = Depends(get_db)
):
    """
    Get a specific compliance report.
    
    Args:
        report_id: The report ID
        format: Output format - "json" or "markdown"
    """
    if not db:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    report = db.query(ComplianceReportModel)\
        .filter(ComplianceReportModel.id == report_id)\
        .first()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report not found: {report_id}"
        )
    
    # Reconstruct the report object
    findings = json.loads(report.findings_json) if report.findings_json else []
    
    response_data = {
        "id": report.id,
        "title": report.title,
        "created_at": report.created_at.isoformat(),
        "overall_status": report.overall_status,
        "summary": report.summary,
        "findings": findings,
        "query": report.query,
        "statistics": {
            "total_findings": len(findings),
            "compliant_count": sum(1 for f in findings if f.get("status") == "compliant"),
            "non_compliant_count": sum(1 for f in findings if f.get("status") == "non_compliant"),
            "documents_referenced": len(report.document_ids) if report.document_ids else 0,
            "images_analyzed": len(report.image_ids) if report.image_ids else 0
        }
    }
    
    if format == "markdown":
        # Generate markdown report
        md_lines = [
            f"# {response_data['title']}",
            f"**Generated:** {response_data['created_at']}",
            f"**Status:** {response_data['overall_status'].replace('_', ' ').title()}",
            "",
            "## Summary",
            response_data['summary'],
            "",
            "## Findings"
        ]
        
        for i, f in enumerate(findings, 1):
            status_emoji = {"compliant": "✅", "non_compliant": "❌", "partial": "⚠️"}.get(f.get("status"), "•")
            md_lines.append(f"\n### {i}. {f.get('category', 'Finding')} {status_emoji}")
            md_lines.append(f"**Status:** {f.get('status', 'unknown')}")
            md_lines.append(f"**Severity:** {f.get('severity', 'medium')}")
            md_lines.append(f"\n{f.get('description', '')}")
            if f.get('recommendation'):
                md_lines.append(f"\n**Recommendation:** {f.get('recommendation')}")
        
        return JSONResponse(
            content={"markdown": "\n".join(md_lines)},
            media_type="application/json"
        )
    
    return response_data


@router.delete("/report/{report_id}")
async def delete_compliance_report(
    report_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a compliance report.
    """
    if not db:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    report = db.query(ComplianceReportModel)\
        .filter(ComplianceReportModel.id == report_id)\
        .first()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report not found: {report_id}"
        )
    
    db.delete(report)
    db.commit()
    
    return {"message": f"Report {report_id} deleted"}


@router.get("/status-options")
async def get_compliance_status_options():
    """
    Get available compliance status options.
    Useful for frontend dropdowns and filters.
    """
    return {
        "statuses": [
            {"value": "compliant", "label": "Compliant", "color": "green"},
            {"value": "non_compliant", "label": "Non-Compliant", "color": "red"},
            {"value": "partial", "label": "Partially Compliant", "color": "yellow"},
            {"value": "needs_review", "label": "Needs Review", "color": "blue"},
            {"value": "insufficient_data", "label": "Insufficient Data", "color": "gray"}
        ],
        "severities": [
            {"value": "low", "label": "Low", "color": "green"},
            {"value": "medium", "label": "Medium", "color": "yellow"},
            {"value": "high", "label": "High", "color": "orange"},
            {"value": "critical", "label": "Critical", "color": "red"}
        ]
    }
