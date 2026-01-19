/**
 * React Query hooks for data fetching and mutations.
 * Provides caching, loading states, and error handling.
 */
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import toast from "react-hot-toast";
import {
  getDocuments,
  uploadDocument,
  deleteDocument,
  bulkDeleteDocuments,
  getDocumentContent,
  sendChatMessage,
  getChatHistory,
  clearChatHistory,
  getImages,
  uploadImage,
  deleteImage,
} from "../api/client";

// ============================================================================
// Document Hooks
// ============================================================================

/**
 * Hook to fetch all documents
 * Automatically refetches and caches data
 */
export const useDocuments = () => {
  return useQuery({
    queryKey: ["documents"],
    queryFn: getDocuments,
    staleTime: 30000, // Consider data fresh for 30 seconds
    refetchOnWindowFocus: true,
  });
};

/**
 * Hook to upload a document
 * Invalidates document list on success
 */
export const useUploadDocument = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: uploadDocument,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      toast.success(`Document "${data.filename}" uploaded successfully`);
    },
    onError: (error) => {
      toast.error(error.response?.data?.detail || "Failed to upload document");
    },
  });
};

/**
 * Hook to delete a document
 * Invalidates document list on success
 */
export const useDeleteDocument = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: deleteDocument,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      toast.success("Document deleted");
    },
    onError: () => {
      toast.error("Failed to delete document");
    },
  });
};

/**
 * Hook to bulk delete multiple documents
 * Invalidates document list on success
 */
export const useBulkDeleteDocuments = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: bulkDeleteDocuments,
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      toast.success(`${variables.length} document(s) deleted`);
    },
    onError: () => {
      toast.error("Failed to delete documents");
    },
  });
};

/**
 * Hook to fetch document content
 * @param {string} docId - Document UUID
 * @param {boolean} enabled - Whether to fetch
 */
export const useDocumentContent = (docId, enabled = true) => {
  return useQuery({
    queryKey: ["documentContent", docId],
    queryFn: () => getDocumentContent(docId),
    enabled: enabled && !!docId,
    staleTime: 60000, // Cache for 1 minute
  });
};

// ============================================================================
// Chat Hooks
// ============================================================================

/**
 * Hook to send chat messages
 * Does not cache chat responses
 */
export const useChatMutation = () => {
  return useMutation({
    mutationFn: sendChatMessage,
    // Don't cache chat responses
    gcTime: 0,
  });
};

/**
 * Hook to fetch chat history for a user
 * @param {string} userId - User identifier
 * @param {number} limit - Max entries to fetch
 */
export const useChatHistory = (userId, limit = 50) => {
  return useQuery({
    queryKey: ["chatHistory", userId],
    queryFn: () => getChatHistory(userId, limit),
    enabled: !!userId, // Only fetch if userId is provided
    staleTime: 60000, // Consider fresh for 1 minute
  });
};

/**
 * Hook to clear chat history
 * Invalidates chat history cache on success
 */
export const useClearChatHistory = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: clearChatHistory,
    onSuccess: (_, userId) => {
      queryClient.invalidateQueries({ queryKey: ["chatHistory", userId] });
      toast.success("Chat history cleared");
    },
    onError: () => {
      toast.error("Failed to clear chat history");
    },
  });
};

// ============================================================================
// Image Hooks
// ============================================================================

/**
 * Hook to fetch all images
 * Automatically refetches and caches data
 */
export const useImages = () => {
  return useQuery({
    queryKey: ["images"],
    queryFn: () => getImages(),
    staleTime: 30000, // Consider data fresh for 30 seconds
    refetchOnWindowFocus: true,
  });
};

/**
 * Hook to upload an image
 * Invalidates images list on success
 */
export const useUploadImage = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: uploadImage,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["images"] });
      toast.success(`Image "${data.filename}" uploaded`);
    },
    onError: (error) => {
      toast.error(error.response?.data?.detail || "Failed to upload image");
    },
  });
};

/**
 * Hook to delete an image
 * Invalidates images list on success
 */
export const useDeleteImage = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: deleteImage,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["images"] });
      toast.success("Image deleted");
    },
    onError: () => {
      toast.error("Failed to delete image");
    },
  });
};
