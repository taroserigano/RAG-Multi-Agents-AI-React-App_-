/**
 * File drop zone component for document upload.
 */
import { useState, useCallback } from "react";
import { Upload, File, AlertCircle, CheckCircle, Cloud } from "lucide-react";

export default function FileDrop({
  onFileSelect,
  accept = ".pdf,.txt",
  maxSizeMB = 15,
}) {
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);

  const validateFile = (file) => {
    // Check file extension
    const ext = file.name.toLowerCase().match(/\.[^.]*$/)?.[0];
    const allowedExts = accept.split(",").map((e) => e.trim());

    if (!allowedExts.includes(ext)) {
      return `Invalid file type. Allowed: ${accept}`;
    }

    // Check file size
    const maxSize = maxSizeMB * 1024 * 1024;
    if (file.size > maxSize) {
      return `File too large. Maximum size: ${maxSizeMB}MB`;
    }

    return null;
  };

  const handleFile = useCallback(
    (file) => {
      setError(null);

      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        setSelectedFile(null);
        return;
      }

      setSelectedFile(file);
      onFileSelect(file);
    },
    [onFileSelect],
  );

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();

    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  return (
    <div className="w-full">
      <div
        className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-200 ${
          dragActive
            ? "border-violet-500 bg-violet-500/10 scale-[1.01]"
            : error
              ? "border-red-500/50 bg-red-500/10"
              : selectedFile
                ? "border-emerald-500/50 bg-emerald-500/10"
                : "border-[var(--border-subtle)] bg-[var(--bg-secondary)]/30 hover:border-violet-500/50 hover:bg-violet-500/5"
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="file-upload"
          accept={accept}
          onChange={handleChange}
          className="hidden"
        />

        <label htmlFor="file-upload" className="cursor-pointer">
          <div className="flex flex-col items-center">
            {error ? (
              <div className="w-14 h-14 rounded-2xl bg-red-500/20 flex items-center justify-center mb-4">
                <AlertCircle className="h-7 w-7 text-red-400" />
              </div>
            ) : selectedFile ? (
              <div className="w-14 h-14 rounded-2xl bg-emerald-500/20 flex items-center justify-center mb-4">
                <CheckCircle className="h-7 w-7 text-emerald-400" />
              </div>
            ) : (
              <div
                className={`w-14 h-14 rounded-2xl flex items-center justify-center mb-4 transition-all ${
                  dragActive ? "bg-violet-500/30 scale-105" : "bg-violet-500/15"
                }`}
              >
                <Cloud
                  className={`h-7 w-7 transition-colors ${dragActive ? "text-violet-300" : "text-violet-400"}`}
                />
              </div>
            )}

            {selectedFile ? (
              <div className="space-y-2">
                <div className="flex items-center justify-center text-emerald-400">
                  <File className="h-4 w-4 mr-2" />
                  <span className="text-sm font-semibold">
                    {selectedFile.name}
                  </span>
                </div>
                <p className="text-xs text-[var(--text-muted)]">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
                <p className="text-xs text-[var(--text-muted)]">
                  Click or drag to replace
                </p>
              </div>
            ) : (
              <>
                <p className="text-base font-semibold text-[var(--text-primary)] mb-1">
                  {dragActive
                    ? "Drop your file here"
                    : "Drop your file here or click to browse"}
                </p>
                <p className="text-xs text-[var(--text-muted)]">
                  Supported: PDF, TXT (max {maxSizeMB}MB)
                </p>
              </>
            )}
          </div>
        </label>

        {/* Animated border when dragging */}
        {dragActive && (
          <div className="absolute inset-0 rounded-2xl border-2 border-violet-500 pointer-events-none" />
        )}
      </div>

      {error && (
        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-xl">
          <p className="text-xs text-red-400 flex items-center">
            <AlertCircle className="h-3.5 w-3.5 mr-2" />
            {error}
          </p>
        </div>
      )}
    </div>
  );
}
