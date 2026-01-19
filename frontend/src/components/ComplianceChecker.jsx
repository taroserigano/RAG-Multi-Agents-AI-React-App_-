/**
 * ComplianceChecker Component
 *
 * A comprehensive compliance checking interface that combines
 * document analysis with image analysis for thorough compliance assessment.
 */
import { useState, useRef, useEffect } from "react";
import {
  Shield,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Search,
  FileText,
  Image as ImageIcon,
  Download,
  Loader2,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Info,
} from "lucide-react";
import { streamComplianceCheck, getComplianceReport } from "../api/client";

// Status badge styling - theme aware
const statusStyles = {
  compliant: {
    bg: "bg-green-500/10",
    text: "text-green-400",
    border: "border-green-500/20",
    icon: CheckCircle,
    label: "Compliant",
  },
  non_compliant: {
    bg: "bg-red-500/10",
    text: "text-red-400",
    border: "border-red-500/20",
    icon: XCircle,
    label: "Non-Compliant",
  },
  partial: {
    bg: "bg-yellow-500/10",
    text: "text-yellow-400",
    border: "border-yellow-500/20",
    icon: AlertTriangle,
    label: "Partially Compliant",
  },
  needs_review: {
    bg: "bg-blue-500/10",
    text: "text-blue-400",
    border: "border-blue-500/20",
    icon: Search,
    label: "Needs Review",
  },
  insufficient_data: {
    bg: "bg-zinc-500/10",
    text: "text-zinc-400",
    border: "border-zinc-500/20",
    icon: AlertCircle,
    label: "Insufficient Data",
  },
};

const severityStyles = {
  low: "text-green-400",
  medium: "text-yellow-400",
  high: "text-orange-400",
  critical: "text-red-400",
};

// Status Badge component
function StatusBadge({ status }) {
  const style = statusStyles[status] || statusStyles.needs_review;
  const Icon = style.icon;

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${style.bg} ${style.text} ${style.border} border`}
    >
      <Icon className="w-3 h-3" />
      {style.label}
    </span>
  );
}

// Finding Card component
function FindingCard({ finding, index }) {
  const [expanded, setExpanded] = useState(false);
  const style = statusStyles[finding.status] || statusStyles.needs_review;

  return (
    <div className={`border rounded-lg p-4 ${style.border} ${style.bg}`}>
      <div
        className="flex items-start justify-between cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-start gap-3">
          <span className="text-[var(--text-muted)] font-mono text-sm">
            {index + 1}.
          </span>
          <div>
            <h4 className="font-medium text-[var(--text-primary)]">
              {finding.category}
            </h4>
            <div className="flex items-center gap-2 mt-1">
              <StatusBadge status={finding.status} />
              <span
                className={`text-xs font-medium ${severityStyles[finding.severity]}`}
              >
                {finding.severity?.toUpperCase()}
              </span>
            </div>
          </div>
        </div>
        {expanded ? (
          <ChevronUp className="w-5 h-5 text-[var(--text-muted)]" />
        ) : (
          <ChevronDown className="w-5 h-5 text-[var(--text-muted)]" />
        )}
      </div>

      {expanded && (
        <div className="mt-4 space-y-3 border-t border-[var(--border-subtle)] pt-3">
          <p className="text-[var(--text-secondary)] text-sm">
            {finding.description}
          </p>

          {finding.policy_reference && (
            <div className="flex items-start gap-2 text-sm">
              <FileText className="w-4 h-4 text-[var(--text-muted)] mt-0.5" />
              <span className="text-[var(--text-secondary)]">
                <strong>Policy:</strong> {finding.policy_reference}
              </span>
            </div>
          )}

          {finding.image_reference && (
            <div className="flex items-start gap-2 text-sm">
              <ImageIcon className="w-4 h-4 text-[var(--text-muted)] mt-0.5" />
              <span className="text-[var(--text-secondary)]">
                <strong>Image:</strong> {finding.image_reference}
              </span>
            </div>
          )}

          {finding.recommendation && (
            <div className="bg-blue-500/10 border border-blue-500/20 rounded p-3 text-sm">
              <div className="flex items-start gap-2">
                <Info className="w-4 h-4 text-blue-400 mt-0.5" />
                <div>
                  <strong className="text-blue-400">Recommendation:</strong>
                  <p className="text-blue-300 mt-1">{finding.recommendation}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Main ComplianceChecker component
export default function ComplianceChecker({
  documents = [],
  images = [],
  selectedDocIds = [],
  selectedImageIds = [],
  provider = "openai",
  userId = "default-user",
  onSelectDocuments,
  onSelectImages,
}) {
  const [query, setQuery] = useState("");
  const [isChecking, setIsChecking] = useState(false);
  const [status, setStatus] = useState(null);
  const [report, setReport] = useState(null);
  const [streamingText, setStreamingText] = useState("");
  const [citations, setCitations] = useState({
    document_citations: [],
    image_citations: [],
  });
  const [error, setError] = useState(null);
  const [showSourceSelection, setShowSourceSelection] = useState(false);

  const abortRef = useRef(null);
  const resultRef = useRef(null);

  // Example compliance queries
  const exampleQueries = [
    "Does this workspace comply with fire safety regulations?",
    "Check if the safety equipment shown meets OSHA requirements",
    "Is this building permit application compliant with zoning laws?",
    "Verify workplace ergonomics against company health policy",
  ];

  const handleCheck = async () => {
    if (!query.trim()) return;

    setIsChecking(true);
    setError(null);
    setReport(null);
    setStreamingText("");
    setCitations({ document_citations: [], image_citations: [] });
    setStatus({
      stage: "starting",
      message: "Initializing compliance check...",
    });

    abortRef.current = streamComplianceCheck(
      {
        user_id: userId,
        query: query.trim(),
        provider,
        doc_ids: selectedDocIds.length > 0 ? selectedDocIds : undefined,
        image_ids: selectedImageIds.length > 0 ? selectedImageIds : undefined,
        include_image_search: true,
      },
      {
        onStatus: (data) => setStatus(data),
        onCitations: (data) => setCitations(data),
        onToken: (token) => setStreamingText((prev) => prev + token),
        onReport: (data) => {
          setReport(data);
          setIsChecking(false);
          // Scroll to results
          setTimeout(() => {
            resultRef.current?.scrollIntoView({ behavior: "smooth" });
          }, 100);
        },
        onDone: () => setIsChecking(false),
        onError: (err) => {
          setError(err.message);
          setIsChecking(false);
        },
      },
    );
  };

  const handleCancel = () => {
    if (abortRef.current) {
      abortRef.current();
      setIsChecking(false);
      setStatus(null);
    }
  };

  const handleDownloadReport = async () => {
    if (!report) return;

    try {
      // Generate markdown from the report data
      const markdown = `# Compliance Analysis Report

**Generated:** ${report.created_at ? new Date(report.created_at).toLocaleString() : new Date().toLocaleString()}
**Status:** ${(report.overall_status || report.status || "Unknown").toUpperCase()}

## Query
${report.query || "N/A"}

## Summary
${report.summary || report.answer || "No summary available"}

## Statistics
- Total Findings: ${report.statistics?.total_findings || 0}
- Compliant: ${report.statistics?.compliant_count || 0}
- Non-Compliant: ${report.statistics?.non_compliant_count || 0}
- Partial: ${report.statistics?.partial_count || 0}

## Documents Referenced
${report.document_citations?.map((d) => `- ${d.filename}`).join("\n") || "None"}

## Images Analyzed
${report.image_citations?.map((i) => `- ${i.filename}`).join("\n") || "None"}
`;

      const blob = new Blob([markdown], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `compliance-report-${report.id || Date.now()}.md`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to download report:", err);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-[var(--border-subtle)] bg-gradient-to-r from-violet-500/10 to-indigo-500/10 p-4">
        <div className="flex items-center gap-2 mb-2">
          <Shield className="w-6 h-6 text-violet-400" />
          <h2 className="text-xl font-semibold text-[var(--text-primary)]">
            Compliance Checker
          </h2>
        </div>
        <p className="text-sm text-[var(--text-secondary)]">
          Analyze images against policy documents to verify compliance
        </p>
      </div>

      {/* Query Input */}
      <div className="p-4 border-b border-[var(--border-subtle)]">
        <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">
          Compliance Question
        </label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g., Does this workspace photo comply with our fire safety policy?"
          className="w-full p-3 border border-[var(--border-subtle)] rounded-lg resize-none focus:ring-2 focus:ring-violet-500 focus:border-violet-500 bg-[var(--input-bg)] text-[var(--text-primary)] placeholder-[var(--text-muted)] transition-colors"
          rows={3}
          disabled={isChecking}
        />

        {/* Example queries */}
        <div className="mt-2">
          <span className="text-xs text-[var(--text-muted)]">Try: </span>
          {exampleQueries.slice(0, 2).map((q, i) => (
            <button
              key={i}
              onClick={() => setQuery(q)}
              className="text-xs text-violet-400 hover:text-violet-300 mr-2"
            >
              "{q.substring(0, 40)}..."
            </button>
          ))}
        </div>
      </div>

      {/* Source Selection Toggle */}
      <div className="p-4 border-b border-[var(--border-subtle)]">
        <button
          onClick={() => setShowSourceSelection(!showSourceSelection)}
          className="flex items-center gap-2 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
        >
          {showSourceSelection ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
          <span>
            Select specific documents & images (
            {selectedDocIds.length + selectedImageIds.length} selected)
          </span>
        </button>

        {showSourceSelection && (
          <div className="mt-3 grid grid-cols-2 gap-4">
            {/* Documents */}
            <div>
              <h4 className="text-sm font-medium text-[var(--text-secondary)] mb-2 flex items-center gap-1">
                <FileText className="w-4 h-4" /> Documents
              </h4>
              <div className="max-h-40 overflow-y-auto border border-[var(--border-subtle)] rounded p-2 space-y-1 bg-[var(--bg-secondary)]">
                {documents.length === 0 ? (
                  <p className="text-xs text-[var(--text-muted)]">
                    No documents uploaded
                  </p>
                ) : (
                  documents.map((doc) => (
                    <label
                      key={doc.id}
                      className="flex items-center gap-2 text-sm cursor-pointer hover:bg-[var(--hover-bg)] p-1 rounded text-[var(--text-secondary)]"
                    >
                      <input
                        type="checkbox"
                        checked={selectedDocIds.includes(doc.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            onSelectDocuments?.([...selectedDocIds, doc.id]);
                          } else {
                            onSelectDocuments?.(
                              selectedDocIds.filter((id) => id !== doc.id),
                            );
                          }
                        }}
                        className="rounded text-violet-600"
                      />
                      <span className="truncate">{doc.filename}</span>
                    </label>
                  ))
                )}
              </div>
            </div>

            {/* Images */}
            <div>
              <h4 className="text-sm font-medium text-[var(--text-secondary)] mb-2 flex items-center gap-1">
                <ImageIcon className="w-4 h-4" /> Images
              </h4>
              <div className="max-h-40 overflow-y-auto border border-[var(--border-subtle)] rounded p-2 space-y-1 bg-[var(--bg-secondary)]">
                {images.length === 0 ? (
                  <p className="text-xs text-[var(--text-muted)]">
                    No images uploaded
                  </p>
                ) : (
                  images.map((img) => (
                    <label
                      key={img.id}
                      className="flex items-center gap-2 text-sm cursor-pointer hover:bg-[var(--hover-bg)] p-1 rounded text-[var(--text-secondary)]"
                    >
                      <input
                        type="checkbox"
                        checked={selectedImageIds.includes(img.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            onSelectImages?.([...selectedImageIds, img.id]);
                          } else {
                            onSelectImages?.(
                              selectedImageIds.filter((id) => id !== img.id),
                            );
                          }
                        }}
                        className="rounded text-violet-600"
                      />
                      {img.thumbnail_base64 && (
                        <img
                          src={`data:image/png;base64,${img.thumbnail_base64}`}
                          alt=""
                          className="w-6 h-6 object-cover rounded"
                        />
                      )}
                      <span className="truncate">{img.filename}</span>
                    </label>
                  ))
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="p-4 border-b border-[var(--border-subtle)] flex gap-2">
        <button
          onClick={handleCheck}
          disabled={isChecking || !query.trim()}
          className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isChecking ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Checking...
            </>
          ) : (
            <>
              <Shield className="w-4 h-4" />
              Run Compliance Check
            </>
          )}
        </button>

        {isChecking && (
          <button
            onClick={handleCancel}
            className="px-4 py-2 border border-[var(--border-subtle)] text-[var(--text-secondary)] rounded-lg hover:bg-[var(--hover-bg)] transition-colors"
          >
            Cancel
          </button>
        )}
      </div>

      {/* Results Area */}
      <div ref={resultRef} className="flex-1 overflow-y-auto p-4">
        {/* Status indicator */}
        {isChecking && status && (
          <div className="mb-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
            <div className="flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
              <span className="text-sm text-blue-400">{status.message}</span>
            </div>
          </div>
        )}

        {/* Error display */}
        {error && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <div className="flex items-center gap-2">
              <XCircle className="w-4 h-4 text-red-400" />
              <span className="text-sm text-red-400">{error}</span>
            </div>
          </div>
        )}

        {/* Streaming analysis text */}
        {isChecking && streamingText && (
          <div className="mb-4 p-4 bg-[var(--bg-secondary)] border border-[var(--border-subtle)] rounded-lg">
            <h4 className="text-sm font-medium text-[var(--text-secondary)] mb-2">
              Analysis in progress...
            </h4>
            <pre className="text-xs text-[var(--text-muted)] whitespace-pre-wrap font-mono max-h-40 overflow-y-auto">
              {streamingText}
            </pre>
          </div>
        )}

        {/* Final Report */}
        {report && (
          <div className="space-y-6">
            {/* Report Header */}
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-lg font-semibold text-[var(--text-primary)]">
                  {report.title || "Compliance Report"}
                </h3>
                <p className="text-sm text-[var(--text-muted)]">
                  Generated:{" "}
                  {report.created_at
                    ? new Date(report.created_at).toLocaleString()
                    : new Date().toLocaleString()}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <StatusBadge status={report.overall_status || report.status} />
                <button
                  onClick={handleDownloadReport}
                  className="p-2 text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--hover-bg)] rounded transition-colors"
                  title="Download report"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Summary */}
            <div className="p-4 bg-[var(--bg-secondary)] border border-[var(--border-subtle)] rounded-lg">
              <h4 className="text-sm font-medium text-[var(--text-secondary)] mb-2">
                Summary
              </h4>
              <p className="text-[var(--text-primary)]">{report.summary}</p>
            </div>

            {/* Statistics */}
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center p-3 bg-[var(--bg-secondary)] rounded-lg border border-[var(--border-subtle)]">
                <div className="text-2xl font-bold text-[var(--text-primary)]">
                  {report.statistics?.total_findings || 0}
                </div>
                <div className="text-xs text-[var(--text-muted)]">
                  Total Findings
                </div>
              </div>
              <div className="text-center p-3 bg-green-500/10 rounded-lg border border-green-500/20">
                <div className="text-2xl font-bold text-green-400">
                  {report.statistics?.compliant_count || 0}
                </div>
                <div className="text-xs text-[var(--text-muted)]">
                  Compliant
                </div>
              </div>
              <div className="text-center p-3 bg-red-500/10 rounded-lg border border-red-500/20">
                <div className="text-2xl font-bold text-red-400">
                  {report.statistics?.non_compliant_count || 0}
                </div>
                <div className="text-xs text-[var(--text-muted)]">
                  Non-Compliant
                </div>
              </div>
              <div className="text-center p-3 bg-yellow-500/10 rounded-lg border border-yellow-500/20">
                <div className="text-2xl font-bold text-yellow-400">
                  {report.statistics?.partial_count || 0}
                </div>
                <div className="text-xs text-[var(--text-muted)]">Partial</div>
              </div>
            </div>

            {/* Findings */}
            {report.findings && report.findings.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-[var(--text-secondary)] mb-3">
                  Detailed Findings
                </h4>
                <div className="space-y-3">
                  {report.findings.map((finding, index) => (
                    <FindingCard
                      key={finding.id || index}
                      finding={finding}
                      index={index}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Sources */}
            <div className="grid grid-cols-2 gap-4">
              {/* Document Citations */}
              {report.document_citations &&
                report.document_citations.length > 0 && (
                  <div className="p-4 border border-[var(--border-subtle)] rounded-lg bg-[var(--bg-secondary)]">
                    <h4 className="text-sm font-medium text-[var(--text-secondary)] mb-2 flex items-center gap-1">
                      <FileText className="w-4 h-4" />
                      Documents Referenced ({report.document_citations.length})
                    </h4>
                    <ul className="space-y-1 text-sm">
                      {report.document_citations.map((doc, i) => (
                        <li
                          key={i}
                          className="flex items-center gap-2 text-[var(--text-secondary)]"
                        >
                          <span className="w-2 h-2 bg-violet-400 rounded-full"></span>
                          {doc.filename}
                          <span className="text-xs text-[var(--text-muted)]">
                            ({Math.round(doc.score * 100)}%)
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

              {/* Image Citations */}
              {report.image_citations && report.image_citations.length > 0 && (
                <div className="p-4 border border-[var(--border-subtle)] rounded-lg bg-[var(--bg-secondary)]">
                  <h4 className="text-sm font-medium text-[var(--text-secondary)] mb-2 flex items-center gap-1">
                    <ImageIcon className="w-4 h-4" />
                    Images Analyzed ({report.image_citations.length})
                  </h4>
                  <div className="grid grid-cols-3 gap-2">
                    {report.image_citations.map((img, i) => (
                      <div key={i} className="text-center">
                        {img.thumbnail_base64 ? (
                          <img
                            src={`data:image/png;base64,${img.thumbnail_base64}`}
                            alt={img.filename}
                            className="w-full h-12 object-cover rounded border border-[var(--border-subtle)]"
                          />
                        ) : (
                          <div className="w-full h-12 bg-[var(--bg-tertiary)] rounded border border-[var(--border-subtle)] flex items-center justify-center">
                            <ImageIcon className="w-4 h-4 text-[var(--text-muted)]" />
                          </div>
                        )}
                        <p className="text-xs text-[var(--text-muted)] truncate mt-1">
                          {img.filename}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Empty state */}
        {!isChecking && !report && !error && (
          <div className="text-center py-12 text-[var(--text-muted)]">
            <Shield className="w-12 h-12 mx-auto mb-4 opacity-30" />
            <p>
              Enter a compliance question above to analyze your documents and
              images
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
