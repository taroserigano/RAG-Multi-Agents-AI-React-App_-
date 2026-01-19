/**
 * ImageGallery component for displaying uploaded images.
 * Supports selection for multimodal chat and deletion.
 */
import { useState } from "react";
import { deleteImage } from "../api/client";

export default function ImageGallery({
  images = [],
  selectedImages = [],
  onSelectionChange,
  onDelete,
  selectable = true,
  showDelete = true,
  loading = false,
}) {
  const [deletingId, setDeletingId] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);

  const handleSelect = (imageId) => {
    if (!selectable) return;

    const newSelection = selectedImages.includes(imageId)
      ? selectedImages.filter((id) => id !== imageId)
      : [...selectedImages, imageId];

    onSelectionChange?.(newSelection);
  };

  const handleDelete = async (imageId, e) => {
    e.stopPropagation();

    if (!confirm("Delete this image?")) return;

    try {
      setDeletingId(imageId);
      await deleteImage(imageId);
      onDelete?.(imageId);
    } catch (error) {
      console.error("Delete failed:", error);
    } finally {
      setDeletingId(null);
    }
  };

  if (loading) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div
            key={i}
            className="aspect-square bg-[var(--bg-secondary)] rounded-lg animate-pulse"
          />
        ))}
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="text-center py-12 text-[var(--text-muted)]">
        <svg
          className="w-12 h-12 mx-auto mb-3 opacity-50"
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
        <p>No images uploaded yet</p>
      </div>
    );
  }

  return (
    <>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
        {images.map((image) => {
          const isSelected = selectedImages.includes(image.id);
          const isDeleting = deletingId === image.id;

          return (
            <div
              key={image.id}
              onClick={() => handleSelect(image.id)}
              className={`
                relative aspect-square rounded-lg overflow-hidden group
                transition-all duration-200
                ${selectable ? "cursor-pointer" : ""}
                ${
                  isSelected
                    ? "ring-2 ring-purple-500 ring-offset-2 ring-offset-[var(--bg-primary)]"
                    : "hover:ring-1 hover:ring-[var(--border-subtle)]"
                }
                ${isDeleting ? "opacity-50" : ""}
              `}
            >
              {/* Image */}
              <img
                src={
                  image.thumbnail_url ||
                  image.url ||
                  (image.thumbnail_base64
                    ? `data:image/png;base64,${image.thumbnail_base64}`
                    : "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect width='100' height='100' fill='%23333'/%3E%3Ctext x='50' y='55' text-anchor='middle' fill='%23666' font-size='12'%3ENo Preview%3C/text%3E%3C/svg%3E")
                }
                alt={image.description || image.filename}
                className="w-full h-full object-cover"
                loading="lazy"
              />

              {/* Selection Indicator */}
              {selectable && (
                <div
                  className={`
                  absolute top-2 left-2 w-6 h-6 rounded-full border-2
                  flex items-center justify-center transition-all
                  ${
                    isSelected
                      ? "bg-purple-500 border-purple-500"
                      : "bg-black/50 border-white/50"
                  }
                `}
                >
                  {isSelected && (
                    <svg
                      className="w-4 h-4 text-white"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                  )}
                </div>
              )}

              {/* Hover Overlay */}
              <div
                className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 
                              transition-opacity flex flex-col justify-end p-3"
              >
                <p className="text-white text-xs font-medium truncate">
                  {image.filename}
                </p>
                {image.description && (
                  <p className="text-[var(--text-secondary)] text-xs truncate mt-1">
                    {image.description}
                  </p>
                )}
              </div>

              {/* Delete Button */}
              {showDelete && (
                <button
                  onClick={(e) => handleDelete(image.id, e)}
                  disabled={isDeleting}
                  className="absolute top-2 right-2 p-1.5 rounded-full
                             bg-red-500/80 hover:bg-red-500 text-white
                             opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  {isDeleting ? (
                    <svg
                      className="w-4 h-4 animate-spin"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                      />
                    </svg>
                  ) : (
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                  )}
                </button>
              )}

              {/* Preview Button */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setPreviewImage(image);
                }}
                className="absolute bottom-2 right-2 p-1.5 rounded-full
                           bg-[var(--bg-secondary)]/80 hover:bg-[var(--bg-secondary)] text-[var(--text-primary)]
                           opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"
                  />
                </svg>
              </button>
            </div>
          );
        })}
      </div>

      {/* Preview Modal */}
      {previewImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
          onClick={() => setPreviewImage(null)}
        >
          <div
            className="relative max-w-4xl max-h-[90vh] bg-[var(--bg-secondary)] rounded-xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={
                previewImage.url ||
                (previewImage.thumbnail_base64
                  ? `data:image/png;base64,${previewImage.thumbnail_base64}`
                  : "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='300'%3E%3Crect width='400' height='300' fill='%23333'/%3E%3Ctext x='200' y='155' text-anchor='middle' fill='%23666' font-size='16'%3ENo Preview Available%3C/text%3E%3C/svg%3E")
              }
              alt={previewImage.description || previewImage.filename}
              className="max-w-full max-h-[80vh] object-contain"
            />

            <div className="p-4 border-t border-[var(--border-subtle)]">
              <h3 className="text-[var(--text-primary)] font-medium">
                {previewImage.filename}
              </h3>
              {previewImage.description && (
                <p className="text-[var(--text-secondary)] text-sm mt-1">
                  {previewImage.description}
                </p>
              )}
              <div className="flex items-center gap-4 mt-2 text-xs text-[var(--text-muted)]">
                <span>
                  {previewImage.width}x{previewImage.height}
                </span>
                <span>{(previewImage.file_size / 1024).toFixed(1)} KB</span>
                <span>{previewImage.format}</span>
              </div>
            </div>

            <button
              onClick={() => setPreviewImage(null)}
              className="absolute top-4 right-4 p-2 rounded-full bg-[var(--bg-secondary)] hover:bg-[var(--hover-bg)] text-[var(--text-primary)]"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>
      )}
    </>
  );
}
