/**
 * Chat page with document filtering and conversation interface.
 * Supports streaming responses and conversation history.
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { AlertCircle, History, Trash2, X } from "lucide-react";
import MessageList from "../components/MessageList";
import ChatBox from "../components/ChatBox";
import DocumentList from "../components/DocumentList";
import ModelPicker from "../components/ModelPicker";
import {
  useDocuments,
  useChatHistory,
  useClearChatHistory,
} from "../hooks/useApi";
import { streamChatMessage } from "../api/client";

// Generate a simple session ID for user tracking
const getUserId = () => {
  let userId = localStorage.getItem("user_id");
  if (!userId) {
    userId = `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem("user_id", userId);
  }
  return userId;
};

export default function ChatPage() {
  // Core state
  const [messages, setMessages] = useState([]);
  const [selectedProvider, setSelectedProvider] = useState("ollama");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedDocIds, setSelectedDocIds] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [showHistory, setShowHistory] = useState(false);

  // DEDICATED STREAMING STATE - These will 100% trigger re-renders
  const [streamingText, setStreamingText] = useState("");
  const [streamingCitations, setStreamingCitations] = useState([]);

  // Refs for cleanup and final values
  const messagesEndRef = useRef(null);
  const abortStreamRef = useRef(null);
  const finalContentRef = useRef("");
  const finalCitationsRef = useRef([]);
  const tokenBufferRef = useRef("");
  const rafIdRef = useRef(null);

  const userId = getUserId();

  const {
    data: documents,
    isLoading: docsLoading,
    error: docsError,
  } = useDocuments();

  const { data: chatHistory, refetch: refetchHistory } = useChatHistory(userId);
  const clearHistoryMutation = useClearChatHistory();

  // Sync streaming text to ref for onDone callback
  useEffect(() => {
    finalContentRef.current = streamingText;
  }, [streamingText]);

  useEffect(() => {
    finalCitationsRef.current = streamingCitations;
  }, [streamingCitations]);

  // Auto-scroll when streaming text changes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  // Cleanup stream on unmount
  useEffect(() => {
    return () => {
      if (abortStreamRef.current) {
        abortStreamRef.current();
      }
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, []);

  const handleProviderChange = (newProvider) => {
    setSelectedProvider(newProvider);
    setSelectedModel("");
  };

  const handleSendMessage = useCallback(
    async (content) => {
      // Add user message
      setMessages((prev) => [...prev, { type: "user", content }]);

      // Reset streaming state
      setStreamingText("");
      setStreamingCitations([]);
      finalContentRef.current = "";
      finalCitationsRef.current = [];
      setIsStreaming(true);

      const chatRequest = {
        user_id: userId,
        provider: selectedProvider,
        model: selectedModel || undefined,
        question: content,
        doc_ids: selectedDocIds.length > 0 ? selectedDocIds : undefined,
        top_k: 5,
      };

      abortStreamRef.current = streamChatMessage(chatRequest, {
        onToken: (token) => {
          // Buffer tokens and batch updates with requestAnimationFrame for performance
          tokenBufferRef.current += token;
          finalContentRef.current += token;

          if (!rafIdRef.current) {
            rafIdRef.current = requestAnimationFrame(() => {
              const buffered = tokenBufferRef.current;
              tokenBufferRef.current = "";
              rafIdRef.current = null;
              setStreamingText((prev) => prev + buffered);
            });
          }
        },
        onCitations: (citations) => {
          setStreamingCitations(citations);
        },
        onDone: (data) => {
          // Cancel any pending animation frame
          if (rafIdRef.current) {
            cancelAnimationFrame(rafIdRef.current);
            rafIdRef.current = null;
          }
          // Flush any remaining buffered tokens
          if (tokenBufferRef.current) {
            finalContentRef.current += tokenBufferRef.current;
            tokenBufferRef.current = "";
          }
          // Add completed message using refs for final content
          setMessages((prev) => [
            ...prev,
            {
              type: "assistant",
              content: finalContentRef.current,
              citations: finalCitationsRef.current,
              model: data.model,
              isStreaming: false,
            },
          ]);
          // Clear streaming state
          setStreamingText("");
          setStreamingCitations([]);
          setIsStreaming(false);
          abortStreamRef.current = null;
          refetchHistory();
        },
        onError: (error) => {
          setMessages((prev) => [
            ...prev,
            {
              type: "assistant",
              content: finalContentRef.current || `Error: ${error.message}`,
              citations: [],
              model: null,
              isStreaming: false,
            },
          ]);
          setStreamingText("");
          setStreamingCitations([]);
          setIsStreaming(false);
          abortStreamRef.current = null;
        },
      });
    },
    [userId, selectedProvider, selectedModel, selectedDocIds, refetchHistory],
  );

  const handleClearHistory = () => {
    if (window.confirm("Are you sure you want to clear all chat history?")) {
      clearHistoryMutation.mutate(userId);
    }
  };

  const loadHistoryEntry = (entry) => {
    setMessages([
      { type: "user", content: entry.question },
      {
        type: "assistant",
        content: entry.answer,
        citations: [],
        model: { provider: entry.provider, name: entry.model },
      },
    ]);
    setShowHistory(false);
  };

  // Build display messages - include streaming message if active
  const displayMessages = isStreaming
    ? [
        ...messages,
        {
          type: "assistant",
          content: streamingText,
          citations: streamingCitations,
          model: null,
          isStreaming: true,
        },
      ]
    : messages;

  return (
    <div className="h-[calc(100vh-8rem)]">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Document Q&A
          </h1>
          <p className="text-gray-600">
            Ask questions about your uploaded documents.
          </p>
        </div>

        <button
          onClick={() => setShowHistory(!showHistory)}
          className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            showHistory
              ? "bg-primary-100 text-primary-700"
              : "text-gray-600 hover:bg-gray-100"
          }`}
        >
          <History className="h-4 w-4 mr-2" />
          History
        </button>
      </div>

      {/* History Panel */}
      {showHistory && (
        <div className="mb-6 bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900">Chat History</h3>
            <div className="flex items-center space-x-2">
              <button
                onClick={handleClearHistory}
                disabled={
                  clearHistoryMutation.isPending || !chatHistory?.length
                }
                className="flex items-center px-3 py-1.5 text-sm text-red-600 hover:bg-red-50 rounded transition-colors disabled:opacity-50"
              >
                <Trash2 className="h-4 w-4 mr-1" />
                Clear All
              </button>
              <button
                onClick={() => setShowHistory(false)}
                className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          {chatHistory && chatHistory.length > 0 ? (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {chatHistory.map((entry) => (
                <button
                  key={entry.id}
                  onClick={() => loadHistoryEntry(entry)}
                  className="w-full text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-md transition-colors"
                >
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {entry.question}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    {new Date(entry.created_at).toLocaleString()} •{" "}
                    {entry.provider}
                  </p>
                </button>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500 text-center py-4">
              No chat history yet
            </p>
          )}
        </div>
      )}

      {/* Model Selection */}
      <div className="mb-6 bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <ModelPicker
          selectedProvider={selectedProvider}
          selectedModel={selectedModel}
          onProviderChange={handleProviderChange}
          onModelChange={setSelectedModel}
        />
      </div>

      {/* Main Chat Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100%-12rem)]">
        {/* Left Sidebar - Document Filter */}
        <div className="lg:col-span-1 bg-white rounded-lg shadow-sm border border-gray-200 p-4 overflow-y-auto">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Filter Documents
          </h2>

          <DocumentList
            documents={documents}
            isLoading={docsLoading}
            error={docsError}
            onSelectionChange={setSelectedDocIds}
          />

          {!docsLoading &&
            !docsError &&
            (!documents || documents.length === 0) && (
              <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
                <p className="text-xs text-yellow-800">
                  No documents available. Upload documents first.
                </p>
              </div>
            )}
        </div>

        {/* Right Side - Chat Interface */}
        <div className="lg:col-span-3 bg-white rounded-lg shadow-sm border border-gray-200 flex flex-col">
          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6">
            {displayMessages.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary-100 mb-4">
                    <AlertCircle className="h-8 w-8 text-primary-600" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    Start a Conversation
                  </h3>
                  <p className="text-gray-600 max-w-sm mx-auto">
                    Ask questions about your uploaded documents.
                  </p>
                </div>
              </div>
            ) : (
              <>
                <MessageList messages={displayMessages} isLoading={false} />
                <div ref={messagesEndRef} />
              </>
            )}
          </div>

          {/* Input Area */}
          <div className="border-t border-gray-200 p-4">
            <ChatBox onSendMessage={handleSendMessage} disabled={isStreaming} />
          </div>
        </div>
      </div>
    </div>
  );
}
