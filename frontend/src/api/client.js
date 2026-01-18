/**
 * API client for backend communication.
 * Uses axios for HTTP requests.
 */
import axios from "axios";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 120000, // 2 minutes for long-running requests
});

// Request interceptor for adding auth headers if needed
api.interceptors.request.use(
  (config) => {
    // Add API key if stored in localStorage
    const apiKey = localStorage.getItem("api_key");
    if (apiKey) {
      config.headers["X-API-Key"] = apiKey;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  },
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(error),
);

// ============================================================================
// Document API
// ============================================================================

/**
 * Upload a document file
 * @param {File} file - File object to upload
 * @returns {Promise} - Upload response with doc_id
 */
export const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post("/api/docs/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return response.data;
};

/**
 * Get list of all documents
 * @returns {Promise} - Array of document metadata
 */
export const getDocuments = async () => {
  const response = await api.get("/api/docs");
  return response.data || [];
};

/**
 * Get specific document by ID
 * @param {string} docId - Document UUID
 * @returns {Promise} - Document metadata
 */
export const getDocument = async (docId) => {
  const response = await api.get(`/api/docs/${docId}`);
  return response.data;
};

/**
 * Delete a document by ID
 * @param {string} docId - Document UUID
 * @returns {Promise} - Delete confirmation
 */
export const deleteDocument = async (docId) => {
  const response = await api.delete(`/api/docs/${docId}`);
  return response.data;
};

/**
 * Bulk delete multiple documents
 * @param {string[]} docIds - Array of document UUIDs
 * @returns {Promise} - Bulk delete result with deleted/failed counts
 */
export const bulkDeleteDocuments = async (docIds) => {
  const response = await api.post("/api/docs/bulk-delete", docIds);
  return response.data;
};

// ============================================================================
// Chat API
// ============================================================================

/**
 * Send a chat question and get RAG-based answer
 * @param {Object} chatRequest - Chat request payload
 * @param {string} chatRequest.user_id - User/session identifier
 * @param {string} chatRequest.provider - LLM provider (ollama/openai/anthropic)
 * @param {string} chatRequest.question - User's question
 * @param {string[]} [chatRequest.doc_ids] - Optional document IDs to filter
 * @param {number} [chatRequest.top_k=5] - Number of chunks to retrieve
 * @param {string} [chatRequest.model] - Optional specific model name
 * @returns {Promise} - Chat response with answer and citations
 */
export const sendChatMessage = async (chatRequest) => {
  const response = await api.post("/api/chat", chatRequest);
  return response.data;
};

/**
 * Stream a chat response using Server-Sent Events
 * @param {Object} chatRequest - Chat request payload
 * @param {Function} onToken - Callback for each token received
 * @param {Function} onCitations - Callback for citations data
 * @param {Function} onDone - Callback when streaming is complete
 * @param {Function} onError - Callback for errors
 * @returns {Function} - Cleanup function to abort the stream
 */
export const streamChatMessage = (
  chatRequest,
  { onToken, onCitations, onDone, onError },
) => {
  const controller = new AbortController();

  const fetchStream = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(chatRequest),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Append decoded chunk to buffer
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        // Process complete lines (SSE events end with double newline)
        const lines = buffer.split("\n\n");

        // Keep the last incomplete chunk in buffer
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data: ")) {
            try {
              const event = JSON.parse(trimmed.slice(6));

              switch (event.type) {
                case "token":
                  onToken?.(event.data);
                  break;
                case "citations":
                  onCitations?.(event.data);
                  break;
                case "done":
                  onDone?.(event.data);
                  break;
                case "error":
                  onError?.(new Error(event.data));
                  break;
              }
            } catch (e) {
              // Skip malformed JSON
            }
          }
        }
      }

      // Process any remaining buffer
      if (buffer.trim().startsWith("data: ")) {
        try {
          const event = JSON.parse(buffer.trim().slice(6));
          if (event.type === "done") onDone?.(event.data);
        } catch (e) {
          // Ignore
        }
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        onError?.(error);
      }
    }
  };

  fetchStream();

  // Return cleanup function
  return () => controller.abort();
};

/**
 * Get chat history for a user
 * @param {string} userId - User identifier
 * @param {number} [limit=50] - Max number of entries to return
 * @returns {Promise} - Array of chat history entries
 */
export const getChatHistory = async (userId, limit = 50) => {
  const response = await api.get(`/api/chat/history/${userId}`, {
    params: { limit },
  });
  return response.data;
};

/**
 * Clear chat history for a user
 * @param {string} userId - User identifier
 * @returns {Promise} - Delete confirmation
 */
export const clearChatHistory = async (userId) => {
  const response = await api.delete(`/api/chat/history/${userId}`);
  return response.data;
};

// ============================================================================
// Health Check
// ============================================================================

/**
 * Check API health status
 * @returns {Promise} - Health status
 */
export const healthCheck = async () => {
  const response = await api.get("/health");
  return response.data;
};

export default api;
