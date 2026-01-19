/**
 * Document list with checkbox selection for filtering.
 */
import { useState } from "react";
import {
  File,
  Loader,
  AlertCircle,
  Trash2,
  CheckCircle2,
  Eye,
} from "lucide-react";
import { useDeleteDocument, useBulkDeleteDocuments } from "../hooks/useApi";
import DocumentPreview from "./DocumentPreview";
import { DocumentListSkeleton } from "./Skeleton";

export default function DocumentList({
  documents,
  isLoading,
  error,
  onSelectionChange,
  showBulkDelete = false,
}) {
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [previewDoc, setPreviewDoc] = useState(null);
  const deleteDocumentMutation = useDeleteDocument();
  const bulkDeleteMutation = useBulkDeleteDocuments();

  const handleToggle = (docId) => {
    const newSelected = new Set(selectedIds);

    if (newSelected.has(docId)) {
      newSelected.delete(docId);
    } else {
      newSelected.add(docId);
    }

    setSelectedIds(newSelected);
    onSelectionChange(Array.from(newSelected));
  };

  const handleSelectAll = () => {
    if (selectedIds.size === documents.length) {
      // Deselect all
      setSelectedIds(new Set());
      onSelectionChange([]);
    } else {
      // Select all
      const allIds = new Set(documents.map((doc) => doc.id));
      setSelectedIds(allIds);
      onSelectionChange(Array.from(allIds));
    }
  };

  const handleDelete = (e, docId) => {
    e.preventDefault();
    e.stopPropagation();

    if (window.confirm("Are you sure you want to delete this document?")) {
      deleteDocumentMutation.mutate(docId, {
        onSuccess: () => {
          // Remove from selected if it was selected
          if (selectedIds.has(docId)) {
            const newSelected = new Set(selectedIds);
            newSelected.delete(docId);
            setSelectedIds(newSelected);
            onSelectionChange(Array.from(newSelected));
          }
        },
      });
    }
  };

  const handlePreview = (e, doc) => {
    e.preventDefault();
    e.stopPropagation();
    setPreviewDoc(doc);
  };

  const handleBulkDelete = () => {
    if (selectedIds.size === 0) return;

    const count = selectedIds.size;
    if (
      window.confirm(
        `Are you sure you want to delete ${count} document${count !== 1 ? "s" : ""}?`,
      )
    ) {
      bulkDeleteMutation.mutate(Array.from(selectedIds), {
        onSuccess: () => {
          setSelectedIds(new Set());
          onSelectionChange([]);
        },
      });
    }
  };

  if (isLoading) {
    return <DocumentListSkeleton count={4} />;
  }

  if (error) {
    return (
      <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl">
        <div className="flex items-start">
          <AlertCircle className="h-4 w-4 text-red-400 mr-2 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-red-400">
              Error loading documents
            </p>
            <p className="text-xs text-red-400/70 mt-1">{error.message}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!documents || documents.length === 0) {
    return (
      <div className="text-center p-8">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-[var(--bg-secondary)]/50 mb-3">
          <File className="h-6 w-6 text-[var(--text-muted)]" />
        </div>
        <p className="text-sm text-[var(--text-secondary)]">
          No documents uploaded yet
        </p>
        <p className="text-xs text-[var(--text-muted)] mt-1">
          Upload documents to get started
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Select All Button */}
      <div className="flex items-center justify-between">
        <button
          onClick={handleSelectAll}
          className="text-left px-3 py-1.5 text-xs font-medium text-violet-400 hover:bg-violet-500/10 rounded-lg transition-colors"
        >
          {selectedIds.size === documents.length
            ? "Deselect All"
            : "Select All"}
        </button>

        {/* Bulk Delete Button */}
        {showBulkDelete && selectedIds.size > 0 && (
          <button
            onClick={handleBulkDelete}
            disabled={bulkDeleteMutation.isPending}
            className="flex items-center px-3 py-1.5 text-xs font-medium text-red-400 hover:bg-red-500/10 rounded-lg transition-colors disabled:opacity-50"
          >
            <Trash2 className="h-3.5 w-3.5 mr-1" />
            Delete ({selectedIds.size})
          </button>
        )}
      </div>

      {/* Document List */}
      <div className="space-y-2">
        {documents.map((doc, index) => {
          const isSelected = selectedIds.has(doc.id);
          return (
            <div
              key={doc.id}
              className={`flex items-start p-3 rounded-xl border transition-all duration-200 group cursor-pointer ${
                isSelected
                  ? "bg-amber-500/10 border-amber-500/30"
                  : "bg-[var(--bg-secondary)]/50 border-[var(--border-subtle)] hover:bg-[var(--hover-bg)] hover:border-[var(--border-subtle)]"
              }`}
              onClick={() => handleToggle(doc.id)}
              style={{ animationDelay: `${index * 50}ms` }}
            >
              {/* Custom Checkbox */}
              <div
                className={`w-4 h-4 rounded-md border-2 flex items-center justify-center transition-all flex-shrink-0 mt-0.5 ${
                  isSelected
                    ? "bg-gradient-to-br from-amber-500 to-orange-500 border-transparent"
                    : "border-[var(--border-subtle)] group-hover:border-amber-500/50"
                }`}
              >
                {isSelected && (
                  <svg
                    className="w-2.5 h-2.5 text-white"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={3}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                )}
              </div>

              <div className="ml-3 flex-1 min-w-0">
                <div className="flex items-start">
                  <File
                    className={`h-3.5 w-3.5 mr-2 mt-0.5 flex-shrink-0 ${isSelected ? "text-amber-400" : "text-[var(--text-muted)]"}`}
                  />
                  <div className="flex-1 min-w-0">
                    <p
                      className={`text-sm font-medium truncate ${isSelected ? "text-[var(--text-primary)]" : "text-[var(--text-secondary)]"}`}
                    >
                      {doc.filename}
                    </p>
                    <p className="text-xs text-[var(--text-muted)] mt-0.5">
                      {new Date(doc.created_at).toLocaleDateString()}
                    </p>
                    {doc.preview_text && (
                      <p className="text-xs text-[var(--text-muted)] mt-1 line-clamp-2">
                        {doc.preview_text}
                      </p>
                    )}
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="ml-2 flex items-center gap-1">
                {/* Preview Button */}
                <button
                  onClick={(e) => handlePreview(e, doc)}
                  className="p-1.5 text-[var(--text-muted)] hover:text-violet-400 hover:bg-violet-500/10 rounded-lg opacity-0 group-hover:opacity-100 transition-all"
                  title="Preview document"
                >
                  <Eye className="h-3.5 w-3.5" />
                </button>

                {/* Delete Button */}
                <button
                  onClick={(e) => handleDelete(e, doc.id)}
                  disabled={deleteDocumentMutation.isPending}
                  className="p-1.5 text-[var(--text-muted)] hover:text-red-400 hover:bg-red-500/10 rounded-lg opacity-0 group-hover:opacity-100 transition-all disabled:opacity-50"
                  title="Delete document"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Selection Info */}
      {selectedIds.size > 0 && (
        <div className="mt-3 p-2.5 bg-amber-500/10 border border-amber-500/20 rounded-xl">
          <p className="text-xs text-amber-400 text-center flex items-center justify-center">
            <CheckCircle2 className="h-3 w-3 mr-1.5" />
            {selectedIds.size} document{selectedIds.size !== 1 ? "s" : ""}{" "}
            selected for filtering
          </p>
        </div>
      )}

      {/* Document Preview Modal */}
      <DocumentPreview
        docId={previewDoc?.id}
        filename={previewDoc?.filename}
        isOpen={!!previewDoc}
        onClose={() => setPreviewDoc(null)}
      />
    </div>
  );
}
