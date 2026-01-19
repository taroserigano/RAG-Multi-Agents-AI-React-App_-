/**
 * ImageUpload component for multimodal RAG.
 * Supports drag & drop and file selection for images.
 */
import { useState, useCallback } from "react";
import { uploadImage } from "../api/client";

const ACCEPTED_FORMATS = [
  "image/png",
  "image/jpeg",
  "image/jpg",
  "image/gif",
  "image/webp",
];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

export default function ImageUpload({ onUploadSuccess, onUploadError }) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [description, setDescription] = useState("");
  const [autoDescribe, setAutoDescribe] = useState(false); // Default to manual description

  const validateFile = (file) => {
    if (!ACCEPTED_FORMATS.includes(file.type)) {
      throw new Error(`Invalid file type. Supported: PNG, JPEG, GIF, WebP`);
    }
    if (file.size > MAX_FILE_SIZE) {
      throw new Error(`File too large. Maximum size: 10MB`);
    }
    return true;
  };

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, []);

  const handleFileSelect = async (file) => {
    try {
      validateFile(file);

      // Create preview and store file for later upload
      const reader = new FileReader();
      reader.onload = (e) => setPreviewUrl(e.target.result);
      reader.readAsDataURL(file);
      setSelectedFile(file);

      // If auto-describe is on, upload immediately
      if (autoDescribe) {
        await performUpload(file, "", true);
      }
    } catch (error) {
      setPreviewUrl(null);
      setSelectedFile(null);
      onUploadError?.(error.message || "File validation failed");
    }
  };

  const performUpload = async (file, desc, autoDesc) => {
    try {
      setIsUploading(true);
      setUploadProgress(20);

      const result = await uploadImage(file, desc, autoDesc);

      setUploadProgress(100);
      setIsUploading(false);
      setPreviewUrl(null);
      setSelectedFile(null);
      setDescription("");

      onUploadSuccess?.(result);
    } catch (error) {
      setIsUploading(false);
      onUploadError?.(error.message || "Upload failed");
    }
  };

  const handleManualUpload = () => {
    if (selectedFile) {
      performUpload(selectedFile, description, false);
    }
  };

  const handleCancelPreview = () => {
    setPreviewUrl(null);
    setSelectedFile(null);
    setDescription("");
  };

  const handleInputChange = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  return (
    <div className="space-y-4">
      {/* Options - shown at top when no file selected */}
      {!selectedFile && (
        <div className="flex items-center justify-between px-2">
          <label className="flex items-center space-x-2 text-sm text-[var(--text-secondary)] cursor-pointer">
            <input
              type="checkbox"
              checked={autoDescribe}
              onChange={(e) => setAutoDescribe(e.target.checked)}
              className="w-4 h-4 rounded border-[var(--border-subtle)] bg-[var(--bg-secondary)] text-purple-500 
                         focus:ring-purple-500 focus:ring-offset-[var(--bg-primary)]"
            />
            <span>Auto-generate description (AI)</span>
          </label>
          {!autoDescribe && (
            <span className="text-xs text-amber-400">
              Manual description mode
            </span>
          )}
        </div>
      )}

      {/* Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          relative border-2 border-dashed rounded-xl p-8 text-center
          transition-all duration-200 cursor-pointer
          ${
            isDragging
              ? "border-purple-500 bg-purple-500/10"
              : "border-[var(--border-subtle)] hover:border-[var(--text-muted)] hover:bg-[var(--bg-secondary)]/30"
          }
          ${isUploading || (selectedFile && !autoDescribe) ? "pointer-events-none" : ""}
          ${isUploading ? "opacity-60" : ""}
        `}
      >
        {!selectedFile && (
          <input
            type="file"
            accept={ACCEPTED_FORMATS.join(",")}
            onChange={handleInputChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            disabled={isUploading}
          />
        )}

        {previewUrl ? (
          <div className="space-y-4">
            <img
              src={previewUrl}
              alt="Preview"
              className="max-h-48 mx-auto rounded-lg shadow-lg"
            />
            {isUploading && (
              <div className="w-full bg-[var(--bg-secondary)] rounded-full h-2">
                <div
                  className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            )}
            {isUploading && (
              <p className="text-sm text-purple-400">
                {autoDescribe ? "Generating AI description..." : "Uploading..."}
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            {/* Image Icon */}
            <div className="flex justify-center">
              <svg
                className={`w-16 h-16 ${isDragging ? "text-purple-400" : "text-[var(--text-muted)]"}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
            </div>

            <div>
              <p className="text-[var(--text-secondary)] font-medium">
                {isDragging ? "Drop image here" : "Drag & drop an image"}
              </p>
              <p className="text-[var(--text-muted)] text-sm mt-1">
                or click to browse
              </p>
            </div>

            <p className="text-xs text-[var(--text-muted)]">
              PNG, JPEG, GIF, WebP • Max 10MB
            </p>
          </div>
        )}
      </div>

      {/* Manual Description Input - shown when file selected and auto-describe is off */}
      {selectedFile && !autoDescribe && !isUploading && (
        <div className="space-y-3 p-4 bg-[var(--bg-secondary)]/50 rounded-lg border border-[var(--border-subtle)]">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-[var(--text-secondary)]">
              Enter description for this image
            </label>
            <button
              onClick={handleCancelPreview}
              className="text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
            >
              Cancel
            </button>
          </div>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Describe what's in this image (optional)..."
            className="w-full px-4 py-3 bg-[var(--bg-secondary)]/50 border border-[var(--border-subtle)] rounded-lg
                       text-[var(--text-primary)] placeholder-[var(--text-muted)] resize-none
                       focus:outline-none focus:ring-2 focus:ring-purple-500/50"
            rows={3}
            autoFocus
          />
          <div className="flex gap-2">
            <button
              onClick={handleManualUpload}
              className="flex-1 px-4 py-2 rounded-lg font-medium transition-all
                bg-purple-600 hover:bg-purple-500 text-white"
            >
              {description.trim()
                ? "Upload with Description"
                : "Upload without Description"}
            </button>
            <button
              onClick={() => performUpload(selectedFile, "", true)}
              className="px-4 py-2 rounded-lg font-medium bg-[var(--bg-secondary)] hover:bg-[var(--hover-bg)] 
                         text-[var(--text-secondary)] transition-all border border-[var(--border-subtle)]"
            >
              Use AI Instead
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
