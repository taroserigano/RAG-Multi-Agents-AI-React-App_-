/**
 * Document list with checkbox selection for filtering.
 */
import { useState } from "react";
import { File, Loader, AlertCircle, Trash2 } from "lucide-react";
import { useDeleteDocument, useBulkDeleteDocuments } from "../hooks/useApi";

export default function DocumentList({
  documents,
  isLoading,
  error,
  onSelectionChange,
  showBulkDelete = false,
}) {
  const [selectedIds, setSelectedIds] = useState(new Set());
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
    return (
      <div className="flex items-center justify-center p-8">
        <Loader className="h-6 w-6 animate-spin text-gray-400" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
        <div className="flex items-start">
          <AlertCircle className="h-5 w-5 text-red-500 mr-2 flex-shrink-0" />
          <div>
            <p className="text-sm font-medium text-red-800">
              Error loading documents
            </p>
            <p className="text-xs text-red-600 mt-1">{error.message}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!documents || documents.length === 0) {
    return (
      <div className="text-center p-8">
        <File className="h-12 w-12 text-gray-300 mx-auto mb-3" />
        <p className="text-sm text-gray-500">No documents uploaded yet</p>
        <p className="text-xs text-gray-400 mt-1">
          Upload documents to get started
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* Select All Button */}
      <div className="flex items-center justify-between">
        <button
          onClick={handleSelectAll}
          className="text-left px-3 py-2 text-sm font-medium text-primary-700 hover:bg-primary-50 rounded-md transition-colors"
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
            className="flex items-center px-3 py-2 text-sm font-medium text-red-600 hover:bg-red-50 rounded-md transition-colors disabled:opacity-50"
          >
            <Trash2 className="h-4 w-4 mr-1" />
            Delete ({selectedIds.size})
          </button>
        )}
      </div>

      {/* Document List */}
      <div className="space-y-1">
        {documents.map((doc) => (
          <div
            key={doc.id}
            className="flex items-start p-3 hover:bg-gray-50 rounded-md transition-colors border border-transparent hover:border-gray-200 group"
          >
            <label className="flex items-start flex-1 cursor-pointer">
              <input
                type="checkbox"
                checked={selectedIds.has(doc.id)}
                onChange={() => handleToggle(doc.id)}
                className="mt-1 h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
              />

              <div className="ml-3 flex-1 min-w-0">
                <div className="flex items-start">
                  <File className="h-4 w-4 text-gray-400 mr-2 mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {doc.filename}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {new Date(doc.created_at).toLocaleDateString()}
                    </p>
                    {doc.preview_text && (
                      <p className="text-xs text-gray-400 mt-1 line-clamp-2">
                        {doc.preview_text}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </label>

            {/* Delete Button */}
            <button
              onClick={(e) => handleDelete(e, doc.id)}
              disabled={deleteDocumentMutation.isPending}
              className="ml-2 p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded opacity-0 group-hover:opacity-100 transition-all disabled:opacity-50"
              title="Delete document"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </div>
        ))}
      </div>

      {/* Selection Info */}
      {selectedIds.size > 0 && (
        <div className="mt-3 p-2 bg-primary-50 border border-primary-200 rounded-md">
          <p className="text-xs text-primary-700 text-center">
            {selectedIds.size} document{selectedIds.size !== 1 ? "s" : ""}{" "}
            selected
          </p>
        </div>
      )}
    </div>
  );
}
