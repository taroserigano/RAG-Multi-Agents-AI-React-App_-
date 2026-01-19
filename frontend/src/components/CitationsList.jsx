/**
 * Citations list component showing source documents.
 */
import { useState } from "react";
import { FileText, ExternalLink, ChevronDown, ChevronUp } from "lucide-react";

export default function CitationsList({ citations }) {
  const [expandedIndex, setExpandedIndex] = useState(null);

  if (!citations || citations.length === 0) {
    return null;
  }

  const toggleExpand = (index) => {
    setExpandedIndex(expandedIndex === index ? null : index);
  };

  return (
    <div className="bg-[var(--bg-secondary)]/60 rounded-lg border border-[var(--border-subtle)] p-4 transition-colors">
      <div className="flex items-center mb-3">
        <ExternalLink className="h-4 w-4 text-[var(--text-muted)] mr-2" />
        <h4 className="text-sm font-semibold text-[var(--text-secondary)]">
          Sources ({citations.length})
        </h4>
      </div>

      <div className="space-y-2">
        {citations.map((citation, index) => (
          <div
            key={index}
            className="bg-[var(--bg-secondary)]/50 rounded-md border border-[var(--border-subtle)] hover:bg-[var(--hover-bg)] transition-colors overflow-hidden"
          >
            <button
              onClick={() => toggleExpand(index)}
              className="w-full flex items-start p-3 text-left"
            >
              <FileText className="h-4 w-4 text-[var(--text-muted)] mr-2 mt-0.5 flex-shrink-0" />

              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-[var(--text-primary)] truncate">
                  {citation.filename}
                </p>

                <div className="flex items-center space-x-3 mt-1 text-xs text-[var(--text-muted)]">
                  {citation.page_number && (
                    <span>Page {citation.page_number}</span>
                  )}
                  <span>Chunk {citation.chunk_index}</span>
                  <span className="text-violet-400 font-medium">
                    Score: {(citation.score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {citation.text && (
                <div className="ml-2 flex-shrink-0">
                  {expandedIndex === index ? (
                    <ChevronUp className="h-4 w-4 text-[var(--text-muted)]" />
                  ) : (
                    <ChevronDown className="h-4 w-4 text-[var(--text-muted)]" />
                  )}
                </div>
              )}
            </button>

            {/* Expandable text snippet */}
            {citation.text && expandedIndex === index && (
              <div className="px-3 pb-3 pt-0">
                <p className="text-xs text-[var(--text-secondary)] bg-[var(--bg-primary)]/50 p-2 rounded border border-[var(--border-subtle)] italic">
                  "{citation.text}"
                </p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
