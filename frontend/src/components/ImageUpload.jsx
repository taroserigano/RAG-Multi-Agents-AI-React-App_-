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
  const [description, setDescription] = useState("");
  const [autoDescribe, setAutoDescribe] = useState(true);

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

      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => setPreviewUrl(e.target.result);
      reader.readAsDataURL(file);

      // Upload
      setIsUploading(true);
      setUploadProgress(20);

      const result = await uploadImage(file, description, autoDescribe);

      setUploadProgress(100);
      setIsUploading(false);
      setPreviewUrl(null);
      setDescription("");

      onUploadSuccess?.(result);
    } catch (error) {
      setIsUploading(false);
      setPreviewUrl(null);
      onUploadError?.(error.message || "Upload failed");
    }
  };

  const handleInputChange = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  return (
    <div className="space-y-4">
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
              : "border-gray-600 hover:border-gray-500 hover:bg-gray-800/30"
          }
          ${isUploading ? "pointer-events-none opacity-60" : ""}
        `}
      >
        <input
          type="file"
          accept={ACCEPTED_FORMATS.join(",")}
          onChange={handleInputChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isUploading}
        />

        {previewUrl ? (
          <div className="space-y-4">
            <img
              src={previewUrl}
              alt="Preview"
              className="max-h-48 mx-auto rounded-lg shadow-lg"
            />
            {isUploading && (
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            {/* Image Icon */}
            <div className="flex justify-center">
              <svg
                className={`w-16 h-16 ${isDragging ? "text-purple-400" : "text-gray-500"}`}
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
              <p className="text-gray-300 font-medium">
                {isDragging ? "Drop image here" : "Drag & drop an image"}
              </p>
              <p className="text-gray-500 text-sm mt-1">or click to browse</p>
            </div>

            <p className="text-xs text-gray-600">
              PNG, JPEG, GIF, WebP • Max 10MB
            </p>
          </div>
        )}
      </div>

      {/* Options */}
      <div className="flex items-center justify-between px-2">
        <label className="flex items-center space-x-2 text-sm text-gray-400 cursor-pointer">
          <input
            type="checkbox"
            checked={autoDescribe}
            onChange={(e) => setAutoDescribe(e.target.checked)}
            className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-purple-500 
                       focus:ring-purple-500 focus:ring-offset-gray-900"
          />
          <span>Auto-generate description (AI)</span>
        </label>
      </div>

      {/* Manual Description */}
      {!autoDescribe && (
        <div>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Add a description for this image..."
            className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700 rounded-lg
                       text-white placeholder-gray-500 resize-none
                       focus:outline-none focus:ring-2 focus:ring-purple-500/50"
            rows={2}
          />
        </div>
      )}
    </div>
  );
}
