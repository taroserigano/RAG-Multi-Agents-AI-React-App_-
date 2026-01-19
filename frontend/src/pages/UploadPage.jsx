/**
 * Upload page for document and image management (multimodal).
 */
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Upload,
  CheckCircle,
  AlertCircle,
  ArrowRight,
  Trash2,
  FileText,
  Sparkles,
  Eye,
  Image as ImageIcon,
} from "lucide-react";
import FileDrop from "../components/FileDrop";
import DocumentPreview from "../components/DocumentPreview";
import ImageUpload from "../components/ImageUpload";
import ImageGallery from "../components/ImageGallery";
import {
  useUploadDocument,
  useDocuments,
  useDeleteDocument,
  useBulkDeleteDocuments,
} from "../hooks/useApi";
import { getImages } from "../api/client";

export default function UploadPage() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("documents"); // "documents" | "images"
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [selectedDocs, setSelectedDocs] = useState(new Set());
  const [previewDoc, setPreviewDoc] = useState(null);

  // Image state
  const [images, setImages] = useState([]);
  const [imagesLoading, setImagesLoading] = useState(false);

  const uploadMutation = useUploadDocument();
  const { data: documents } = useDocuments();
  const deleteDocMutation = useDeleteDocument();
  const bulkDeleteMutation = useBulkDeleteDocuments();

  // Load images when tab changes
  useEffect(() => {
    if (activeTab === "images") {
      loadImages();
    }
  }, [activeTab]);

  const loadImages = async () => {
    try {
      setImagesLoading(true);
      const data = await getImages();
      setImages(data);
    } catch (error) {
      console.error("Failed to load images:", error);
    } finally {
      setImagesLoading(false);
    }
  };

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setUploadStatus(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    try {
      setUploadStatus({
        type: "loading",
        message: "Uploading and indexing...",
      });

      const result = await uploadMutation.mutateAsync(selectedFile);

      setUploadStatus({
        type: "success",
        message: `Successfully uploaded: ${result.filename}`,
      });

      // Reset after 2 seconds
      setTimeout(() => {
        setSelectedFile(null);
        setUploadStatus(null);
      }, 2000);
    } catch (error) {
      setUploadStatus({
        type: "error",
        message: error.response?.data?.detail || "Failed to upload document",
      });
    }
  };

  const handleImageUploadSuccess = async (result) => {
    // Reload images to get complete data including thumbnail_base64
    await loadImages();
    setUploadStatus({
      type: "success",
      message: `Image uploaded: ${result.filename}`,
    });
    setTimeout(() => setUploadStatus(null), 3000);
  };

  const handleImageUploadError = (error) => {
    setUploadStatus({
      type: "error",
      message: error,
    });
  };

  const handleImageDelete = (imageId) => {
    setImages((prev) => prev.filter((img) => img.id !== imageId));
  };

  const toggleDocSelection = (docId) => {
    const newSelected = new Set(selectedDocs);
    if (newSelected.has(docId)) {
      newSelected.delete(docId);
    } else {
      newSelected.add(docId);
    }
    setSelectedDocs(newSelected);
  };

  const handleSelectAll = () => {
    if (selectedDocs.size === documents?.length) {
      setSelectedDocs(new Set());
    } else {
      setSelectedDocs(new Set(documents?.map((d) => d.id) || []));
    }
  };

  const handleDeleteSelected = () => {
    if (selectedDocs.size === 0) return;

    const count = selectedDocs.size;
    if (
      window.confirm(
        `Are you sure you want to delete ${count} document${count !== 1 ? "s" : ""}?`,
      )
    ) {
      bulkDeleteMutation.mutate(Array.from(selectedDocs), {
        onSuccess: () => {
          setSelectedDocs(new Set());
        },
      });
    }
  };

  const handleDeleteSingle = (docId, filename) => {
    if (window.confirm(`Are you sure you want to delete "${filename}"?`)) {
      deleteDocMutation.mutate(docId, {
        onSuccess: () => {
          selectedDocs.delete(docId);
          setSelectedDocs(new Set(selectedDocs));
        },
      });
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <div className="flex items-center mb-3">
          <div className="p-3 rounded-xl bg-violet-500/15 mr-4">
            <Upload className="h-6 w-6 text-violet-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-[var(--text-primary)]">
              Upload Content
            </h1>
            <p className="text-[var(--text-muted)] text-sm mt-1">
              Add documents and images to your multimodal knowledge base
            </p>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setActiveTab("documents")}
          className={`flex items-center px-4 py-2.5 rounded-xl font-medium text-sm transition-all ${
            activeTab === "documents"
              ? "bg-violet-500/20 text-violet-400 border border-violet-500/30"
              : "bg-[var(--bg-secondary)] text-[var(--text-secondary)] border border-[var(--border-subtle)] hover:bg-[var(--hover-bg)] hover:text-[var(--text-primary)]"
          }`}
        >
          <FileText className="h-4 w-4 mr-2" />
          Documents
        </button>
        <button
          onClick={() => setActiveTab("images")}
          className={`flex items-center px-4 py-2.5 rounded-xl font-medium text-sm transition-all ${
            activeTab === "images"
              ? "bg-fuchsia-500/20 text-fuchsia-400 border border-fuchsia-500/30"
              : "bg-[var(--bg-secondary)] text-[var(--text-secondary)] border border-[var(--border-subtle)] hover:bg-[var(--hover-bg)] hover:text-[var(--text-primary)]"
          }`}
        >
          <ImageIcon className="h-4 w-4 mr-2" />
          Images
          <span className="ml-2 px-1.5 py-0.5 text-xs bg-fuchsia-500/20 rounded">
            New
          </span>
        </button>
      </div>

      {/* Status Messages (shared) */}
      {uploadStatus && (
        <div
          className={`mb-6 p-4 rounded-xl border ${
            uploadStatus.type === "success"
              ? "bg-emerald-500/10 border-emerald-500/30"
              : uploadStatus.type === "error"
                ? "bg-red-500/10 border-red-500/30"
                : "bg-blue-500/10 border-blue-500/30"
          }`}
        >
          <div className="flex items-center">
            {uploadStatus.type === "success" && (
              <CheckCircle className="h-4 w-4 text-emerald-400 mr-2" />
            )}
            {uploadStatus.type === "error" && (
              <AlertCircle className="h-4 w-4 text-red-400 mr-2" />
            )}
            {uploadStatus.type === "loading" && (
              <div className="h-4 w-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin mr-2" />
            )}
            <p
              className={`text-sm font-medium ${
                uploadStatus.type === "success"
                  ? "text-emerald-400"
                  : uploadStatus.type === "error"
                    ? "text-red-400"
                    : "text-blue-400"
              }`}
            >
              {uploadStatus.message}
            </p>
          </div>
        </div>
      )}

      {/* Documents Tab */}
      {activeTab === "documents" && (
        <>
          {/* Document Upload Section */}
          <div className="bg-[var(--bg-secondary)]/60 backdrop-blur-sm rounded-2xl border border-[var(--border-subtle)] p-6 mb-8 transition-colors">
            <FileDrop onFileSelect={handleFileSelect} />

            {selectedFile && !uploadStatus && (
              <div className="mt-6">
                <button
                  onClick={handleUpload}
                  disabled={uploadMutation.isPending}
                  className="w-full flex items-center justify-center px-6 py-3.5 bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white font-semibold rounded-xl hover:from-violet-500 hover:to-fuchsia-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  <Upload className="h-4 w-4 mr-2" />
                  {uploadMutation.isPending
                    ? "Uploading..."
                    : "Upload & Index Document"}
                </button>
              </div>
            )}
          </div>

          {/* Documents List */}
          <div className="bg-[var(--bg-secondary)]/60 backdrop-blur-sm rounded-2xl border border-[var(--border-subtle)] p-6 transition-colors">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center">
                <div className="p-2 rounded-xl bg-amber-500/15 mr-3">
                  <FileText className="h-5 w-5 text-amber-400" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-[var(--text-primary)]">
                    Uploaded Documents
                  </h2>
                  <p className="text-xs text-[var(--text-muted)]">
                    {documents?.length || 0} documents in knowledge base
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                {selectedDocs.size > 0 && (
                  <button
                    onClick={handleDeleteSelected}
                    disabled={bulkDeleteMutation.isPending}
                    className="flex items-center px-3 py-2 text-xs font-medium text-red-400 bg-red-500/10 hover:bg-red-500/20 rounded-xl transition-colors disabled:opacity-50"
                  >
                    <Trash2 className="h-3.5 w-3.5 mr-1.5" />
                    Delete ({selectedDocs.size})
                  </button>
                )}

                {documents && documents.length > 0 && (
                  <button
                    onClick={() => navigate("/chat")}
                    className="flex items-center px-4 py-2 text-xs font-semibold bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white rounded-xl hover:from-violet-500 hover:to-fuchsia-500 transition-all"
                  >
                    <Sparkles className="h-3.5 w-3.5 mr-1.5" />
                    Start Chat
                    <ArrowRight className="h-3.5 w-3.5 ml-1.5" />
                  </button>
                )}
              </div>
            </div>

            {documents && documents.length > 0 ? (
              <>
                {/* Select All */}
                <button
                  onClick={handleSelectAll}
                  className="mb-4 text-xs font-medium text-violet-400 hover:text-violet-300 transition-colors"
                >
                  {selectedDocs.size === documents.length
                    ? "Deselect All"
                    : "Select All"}
                </button>

                <div className="space-y-2">
                  {documents.map((doc, index) => {
                    const isSelected = selectedDocs.has(doc.id);
                    return (
                      <div
                        key={doc.id}
                        className={`p-4 rounded-xl border transition-all duration-200 group ${
                          isSelected
                            ? "bg-violet-500/10 border-violet-500/30"
                            : "bg-[var(--bg-secondary)]/50 border-[var(--border-subtle)] hover:bg-[var(--hover-bg)] hover:border-[var(--border-subtle)]"
                        }`}
                        style={{ animationDelay: `${index * 50}ms` }}
                      >
                        <div className="flex items-start">
                          {/* Custom Checkbox */}
                          <div
                            onClick={() => toggleDocSelection(doc.id)}
                            className={`w-4 h-4 rounded-md border-2 flex items-center justify-center transition-all flex-shrink-0 mt-0.5 cursor-pointer ${
                              isSelected
                                ? "bg-gradient-to-br from-violet-500 to-fuchsia-500 border-transparent"
                                : "border-[var(--border-subtle)] hover:border-violet-500/50"
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

                          <div className="ml-4 flex-1">
                            <div className="flex items-center">
                              <FileText
                                className={`h-3.5 w-3.5 mr-2 ${isSelected ? "text-violet-400" : "text-[var(--text-muted)]"}`}
                              />
                              <h3
                                className={`text-sm font-medium ${isSelected ? "text-[var(--text-primary)]" : "text-[var(--text-secondary)]"}`}
                              >
                                {doc.filename}
                              </h3>
                            </div>
                            <p className="text-xs text-[var(--text-muted)] mt-1">
                              Uploaded:{" "}
                              {new Date(doc.created_at).toLocaleString()}
                            </p>
                            {doc.preview_text && (
                              <p className="text-xs text-[var(--text-muted)] mt-1.5 line-clamp-2">
                                {doc.preview_text}
                              </p>
                            )}
                          </div>
                          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all">
                            <button
                              onClick={() => setPreviewDoc(doc)}
                              className="p-2 text-[var(--text-muted)] hover:text-violet-400 hover:bg-violet-500/10 rounded-lg transition-all"
                              title="Preview document"
                            >
                              <Eye className="h-3.5 w-3.5" />
                            </button>
                            <button
                              onClick={() =>
                                handleDeleteSingle(doc.id, doc.filename)
                              }
                              disabled={deleteDocMutation.isPending}
                              className="p-2 text-[var(--text-muted)] hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all disabled:opacity-50"
                              title="Delete document"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </>
            ) : (
              <div className="text-center py-12">
                <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-[var(--bg-secondary)]/50 mb-4">
                  <FileText className="h-7 w-7 text-[var(--text-muted)]" />
                </div>
                <p className="text-[var(--text-secondary)]">
                  No documents uploaded yet
                </p>
                <p className="text-xs text-[var(--text-muted)] mt-1">
                  Upload your first document above to get started
                </p>
              </div>
            )}
          </div>

          {/* Document Preview Modal */}
          <DocumentPreview
            docId={previewDoc?.id}
            filename={previewDoc?.filename}
            isOpen={!!previewDoc}
            onClose={() => setPreviewDoc(null)}
          />
        </>
      )}

      {/* Images Tab */}
      {activeTab === "images" && (
        <>
          {/* Image Upload Section */}
          <div className="bg-[var(--bg-secondary)]/60 backdrop-blur-sm rounded-2xl border border-[var(--border-subtle)] p-6 mb-8 transition-colors">
            <div className="flex items-center mb-4">
              <div className="p-2 rounded-xl bg-fuchsia-500/15 mr-3">
                <ImageIcon className="h-5 w-5 text-fuchsia-400" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-[var(--text-primary)]">
                  Upload Image
                </h2>
                <p className="text-xs text-[var(--text-muted)]">
                  Images are indexed with CLIP for semantic search
                </p>
              </div>
            </div>

            <ImageUpload
              onUploadSuccess={handleImageUploadSuccess}
              onUploadError={handleImageUploadError}
            />
          </div>

          {/* Image Gallery */}
          <div className="bg-[var(--bg-secondary)]/60 backdrop-blur-sm rounded-2xl border border-[var(--border-subtle)] p-6 transition-colors">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center">
                <div className="p-2 rounded-xl bg-cyan-500/15 mr-3">
                  <ImageIcon className="h-5 w-5 text-cyan-400" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-[var(--text-primary)]">
                    Image Gallery
                  </h2>
                  <p className="text-xs text-[var(--text-muted)]">
                    {images.length} images in knowledge base
                  </p>
                </div>
              </div>

              {images.length > 0 && (
                <button
                  onClick={() => navigate("/chat")}
                  className="flex items-center px-4 py-2 text-xs font-semibold bg-gradient-to-r from-fuchsia-600 to-cyan-600 text-white rounded-xl hover:from-fuchsia-500 hover:to-cyan-500 transition-all"
                >
                  <Sparkles className="h-3.5 w-3.5 mr-1.5" />
                  Chat with Images
                  <ArrowRight className="h-3.5 w-3.5 ml-1.5" />
                </button>
              )}
            </div>

            <ImageGallery
              images={images}
              onDelete={handleImageDelete}
              loading={imagesLoading}
              selectable={false}
              showDelete={true}
            />
          </div>
        </>
      )}
    </div>
  );
}
