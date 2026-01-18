/**
 * Chat page with document filtering and conversation interface.
 * Supports streaming responses and conversation history.
 */
import { useState, useEffect, useRef, useCallback } from "react";
import {
  AlertCircle,
  History,
  Trash2,
  X,
  Settings2,
  Sparkles,
  Search,
  BarChart3,
  FileText,
  Image,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import MessageList from "../components/MessageList";
import ChatBox from "../components/ChatBox";
import DocumentList from "../components/DocumentList";
import ModelPicker from "../components/ModelPicker";
import ImageGallery from "../components/ImageGallery";
import {
  useDocuments,
  useChatHistory,
  useClearChatHistory,
} from "../hooks/useApi";
import { streamChatMessage, streamMultimodalChat } from "../api/client";

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
  const [selectedImageIds, setSelectedImageIds] = useState([]);
  const [images, setImages] = useState([]);
  const [showImages, setShowImages] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showRagOptions, setShowRagOptions] = useState(false);

  // Advanced RAG options
  const [ragOptions, setRagOptions] = useState({
    query_expansion: false,
    hybrid_search: false,
    reranking: false,
  });

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

  // Load images on mount
  useEffect(() => {
    const loadImages = async () => {
      try {
        const response = await fetch("http://localhost:8000/api/images");
        if (response.ok) {
          const data = await response.json();
          setImages(data);
        }
      } catch (error) {
        console.error("Failed to load images:", error);
      }
    };
    loadImages();
  }, []);

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
      // Add user message (include selected images info)
      const userMessage = {
        type: "user",
        content,
        imageIds:
          selectedImageIds.length > 0 ? [...selectedImageIds] : undefined,
      };
      setMessages((prev) => [...prev, userMessage]);

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
        image_ids: selectedImageIds.length > 0 ? selectedImageIds : undefined,
        top_k: 5,
        rag_options: ragOptions,
      };

      // Use multimodal chat if images are selected, otherwise use regular chat
      const streamFn =
        selectedImageIds.length > 0 ? streamMultimodalChat : streamChatMessage;

      abortStreamRef.current = streamFn(chatRequest, {
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
    [
      userId,
      selectedProvider,
      selectedModel,
      selectedDocIds,
      selectedImageIds,
      ragOptions,
      refetchHistory,
    ],
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

  // Count active RAG options
  const activeRagOptionsCount =
    Object.values(ragOptions).filter(Boolean).length;

  return (
    <div className="min-h-[calc(100vh-8rem)] flex flex-col">
      {/* Header */}
      <div className="mb-8 flex items-center justify-between flex-shrink-0">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1 flex items-center tracking-tight">
            <span className="gradient-text-vibrant">Document Q&A</span>
            <Sparkles className="h-5 w-5 ml-2 text-amber-400" />
          </h1>
          <p className="text-zinc-500 text-sm">
            Ask questions about your uploaded documents
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowRagOptions(!showRagOptions)}
            className={`flex items-center px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
              showRagOptions
                ? "bg-violet-500/15 text-violet-300 border border-violet-500/25"
                : "text-zinc-500 hover:text-white bg-zinc-900/50 border border-zinc-800 hover:border-violet-500/30"
            }`}
          >
            <Settings2
              className={`h-4 w-4 mr-2 ${showRagOptions ? "text-violet-400" : ""}`}
            />
            RAG
            {activeRagOptionsCount > 0 && (
              <span className="ml-2 bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white text-[10px] px-1.5 py-0.5 rounded-full font-bold">
                {activeRagOptionsCount}
              </span>
            )}
          </button>
          <button
            onClick={() => setShowHistory(!showHistory)}
            className={`flex items-center px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
              showHistory
                ? "bg-blue-500/15 text-blue-300 border border-blue-500/25"
                : "text-zinc-500 hover:text-white bg-zinc-900/50 border border-zinc-800 hover:border-blue-500/30"
            }`}
          >
            <History
              className={`h-4 w-4 mr-2 ${showHistory ? "text-blue-400" : ""}`}
            />
            History
          </button>
        </div>
      </div>

      {/* RAG Options Panel */}
      {showRagOptions && (
        <div className="mb-6 bg-zinc-900/60 backdrop-blur-sm rounded-2xl p-5 border border-zinc-800/80 animate-slideUp">
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center">
              <div className="p-2 rounded-xl bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 mr-3">
                <Sparkles className="h-5 w-5 text-violet-400" />
              </div>
              <div>
                <h3 className="font-semibold text-white text-sm">
                  Advanced RAG
                </h3>
                <p className="text-xs text-zinc-600">Enhance search quality</p>
              </div>
            </div>
            <button
              onClick={() => setShowRagOptions(false)}
              className="p-2 text-zinc-600 hover:text-white hover:bg-zinc-800 rounded-lg transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {/* Query Expansion */}
            <label
              className={`relative flex items-start p-4 rounded-xl border cursor-pointer transition-all duration-200 group ${
                ragOptions.query_expansion
                  ? "bg-violet-500/10 border-violet-500/30"
                  : "bg-zinc-800/50 border-zinc-700/50 hover:border-violet-500/30"
              }`}
            >
              <input
                type="checkbox"
                checked={ragOptions.query_expansion}
                onChange={(e) =>
                  setRagOptions((prev) => ({
                    ...prev,
                    query_expansion: e.target.checked,
                  }))
                }
                className="sr-only"
              />
              <div
                className={`w-4 h-4 rounded-md border-2 flex items-center justify-center transition-all flex-shrink-0 ${
                  ragOptions.query_expansion
                    ? "bg-gradient-to-br from-violet-500 to-fuchsia-500 border-transparent"
                    : "border-zinc-600 group-hover:border-violet-500/50"
                }`}
              >
                {ragOptions.query_expansion && (
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
              <div className="ml-3">
                <div className="flex items-center">
                  <Sparkles
                    className={`h-3.5 w-3.5 mr-1.5 ${ragOptions.query_expansion ? "text-violet-400" : "text-zinc-500"}`}
                  />
                  <span
                    className={`text-sm font-medium ${ragOptions.query_expansion ? "text-white" : "text-zinc-400"}`}
                  >
                    Query Expansion
                  </span>
                </div>
                <p className="text-xs text-zinc-600 mt-0.5">
                  Multiple query variations
                </p>
              </div>
            </label>

            {/* Hybrid Search */}
            <label
              className={`relative flex items-start p-4 rounded-xl border cursor-pointer transition-all duration-200 group ${
                ragOptions.hybrid_search
                  ? "bg-blue-500/10 border-blue-500/30"
                  : "bg-zinc-800/50 border-zinc-700/50 hover:border-blue-500/30"
              }`}
            >
              <input
                type="checkbox"
                checked={ragOptions.hybrid_search}
                onChange={(e) =>
                  setRagOptions((prev) => ({
                    ...prev,
                    hybrid_search: e.target.checked,
                  }))
                }
                className="sr-only"
              />
              <div
                className={`w-4 h-4 rounded-md border-2 flex items-center justify-center transition-all flex-shrink-0 ${
                  ragOptions.hybrid_search
                    ? "bg-gradient-to-br from-blue-500 to-cyan-500 border-transparent"
                    : "border-zinc-600 group-hover:border-blue-500/50"
                }`}
              >
                {ragOptions.hybrid_search && (
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
              <div className="ml-3">
                <div className="flex items-center">
                  <Search
                    className={`h-3.5 w-3.5 mr-1.5 ${ragOptions.hybrid_search ? "text-blue-400" : "text-zinc-500"}`}
                  />
                  <span
                    className={`text-sm font-medium ${ragOptions.hybrid_search ? "text-white" : "text-zinc-400"}`}
                  >
                    Hybrid Search
                  </span>
                </div>
                <p className="text-xs text-slate-500 mt-1">
                  Semantic + keyword search
                </p>
              </div>
            </label>

            {/* Reranking */}
            <label
              className={`relative flex items-start p-4 rounded-xl border cursor-pointer transition-all duration-200 group ${
                ragOptions.reranking
                  ? "bg-emerald-500/10 border-emerald-500/30"
                  : "bg-zinc-800/50 border-zinc-700/50 hover:border-emerald-500/30"
              }`}
            >
              <input
                type="checkbox"
                checked={ragOptions.reranking}
                onChange={(e) =>
                  setRagOptions((prev) => ({
                    ...prev,
                    reranking: e.target.checked,
                  }))
                }
                className="sr-only"
              />
              <div
                className={`w-4 h-4 rounded-md border-2 flex items-center justify-center transition-all flex-shrink-0 ${
                  ragOptions.reranking
                    ? "bg-gradient-to-br from-emerald-500 to-teal-500 border-transparent"
                    : "border-zinc-600 group-hover:border-emerald-500/50"
                }`}
              >
                {ragOptions.reranking && (
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
              <div className="ml-3">
                <div className="flex items-center">
                  <BarChart3
                    className={`h-3.5 w-3.5 mr-1.5 ${ragOptions.reranking ? "text-emerald-400" : "text-zinc-500"}`}
                  />
                  <span
                    className={`text-sm font-medium ${ragOptions.reranking ? "text-white" : "text-zinc-400"}`}
                  >
                    Reranking
                  </span>
                </div>
                <p className="text-xs text-zinc-600 mt-0.5">
                  Relevance scoring
                </p>
              </div>
            </label>
          </div>

          {activeRagOptionsCount > 0 && (
            <div className="mt-4 pt-4 border-t border-zinc-800 flex items-center justify-between">
              <p className="text-xs text-violet-400 flex items-center">
                <Sparkles className="h-3 w-3 mr-1.5" />
                {activeRagOptionsCount} option
                {activeRagOptionsCount > 1 ? "s" : ""} enabled
              </p>
              <span className="text-xs text-zinc-600">May take longer</span>
            </div>
          )}
        </div>
      )}

      {/* History Panel */}
      {showHistory && (
        <div className="mb-6 bg-zinc-900/60 backdrop-blur-sm rounded-2xl p-5 border border-zinc-800/80 animate-slideUp">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <div className="p-2 rounded-xl bg-blue-500/15 mr-3">
                <History className="h-5 w-5 text-blue-400" />
              </div>
              <h3 className="font-semibold text-white text-sm">Chat History</h3>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleClearHistory}
                disabled={
                  clearHistoryMutation.isPending || !chatHistory?.length
                }
                className="flex items-center px-3 py-1.5 text-xs text-red-400 hover:bg-red-500/10 rounded-lg transition-colors disabled:opacity-50"
              >
                <Trash2 className="h-4 w-4 mr-1" />
                Clear All
              </button>
              <button
                onClick={() => setShowHistory(false)}
                className="p-2 text-slate-500 hover:text-white hover:bg-white/5 rounded-lg transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          {chatHistory && chatHistory.length > 0 ? (
            <div className="space-y-2 max-h-64 overflow-y-auto custom-scrollbar">
              {chatHistory.map((entry, index) => (
                <button
                  key={entry.id}
                  onClick={() => loadHistoryEntry(entry)}
                  className="w-full text-left p-3 bg-zinc-800/50 hover:bg-zinc-800 rounded-xl border border-zinc-700/50 hover:border-blue-500/30 transition-all duration-200 group"
                  style={{ animationDelay: `${index * 50}ms` }}
                >
                  <p className="text-sm font-medium text-zinc-200 truncate group-hover:text-blue-300 transition-colors">
                    {entry.question}
                  </p>
                  <p className="text-xs text-zinc-600 mt-1">
                    {new Date(entry.created_at).toLocaleString()} •{" "}
                    <span className="text-blue-400/70">{entry.provider}</span>
                  </p>
                </button>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-zinc-800/50 mb-3">
                <History className="h-6 w-6 text-zinc-600" />
              </div>
              <p className="text-sm text-zinc-500">No chat history yet</p>
            </div>
          )}
        </div>
      )}

      {/* Model Selection */}
      <div className="mb-6 bg-zinc-900/60 backdrop-blur-sm rounded-2xl p-5 border border-zinc-800/80">
        <ModelPicker
          selectedProvider={selectedProvider}
          selectedModel={selectedModel}
          onProviderChange={handleProviderChange}
          onModelChange={setSelectedModel}
        />
      </div>

      {/* Main Chat Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 flex-1 min-h-[500px]">
        {/* Left Sidebar - Document Filter */}
        <div className="lg:col-span-1 bg-zinc-900/60 backdrop-blur-sm rounded-2xl border border-zinc-800/80 p-5 overflow-y-auto custom-scrollbar max-h-[600px]">
          <div className="flex items-center mb-4">
            <div className="p-2 rounded-xl bg-amber-500/15 mr-3">
              <FileText className="h-5 w-5 text-amber-400" />
            </div>
            <h2 className="text-sm font-semibold text-white">Documents</h2>
          </div>

          <DocumentList
            documents={documents}
            isLoading={docsLoading}
            error={docsError}
            onSelectionChange={setSelectedDocIds}
          />

          {!docsLoading &&
            !docsError &&
            (!documents || documents.length === 0) && (
              <div className="mt-4 p-4 bg-amber-500/10 border border-amber-500/20 rounded-xl">
                <p className="text-xs text-amber-400">
                  No documents available. Upload documents first.
                </p>
              </div>
            )}

          {/* Images Section */}
          <div className="mt-6 pt-6 border-t border-zinc-800">
            <button
              onClick={() => setShowImages(!showImages)}
              className="w-full flex items-center justify-between mb-3"
            >
              <div className="flex items-center">
                <div className="p-2 rounded-xl bg-fuchsia-500/15 mr-3">
                  <Image className="h-5 w-5 text-fuchsia-400" />
                </div>
                <h2 className="text-sm font-semibold text-white">Images</h2>
                {selectedImageIds.length > 0 && (
                  <span className="ml-2 px-2 py-0.5 text-xs rounded-full bg-fuchsia-500/20 text-fuchsia-400 border border-fuchsia-500/30">
                    {selectedImageIds.length} selected
                  </span>
                )}
              </div>
              {showImages ? (
                <ChevronUp className="h-4 w-4 text-zinc-500" />
              ) : (
                <ChevronDown className="h-4 w-4 text-zinc-500" />
              )}
            </button>

            {showImages && (
              <div className="space-y-3">
                {images.length > 0 ? (
                  <>
                    <p className="text-xs text-zinc-500 mb-2">
                      Select images to ask questions about them
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                      {images.map((img) => (
                        <button
                          key={img.id}
                          onClick={() => {
                            setSelectedImageIds((prev) =>
                              prev.includes(img.id)
                                ? prev.filter((id) => id !== img.id)
                                : [...prev, img.id],
                            );
                          }}
                          className={`relative aspect-square rounded-lg overflow-hidden border-2 transition-all ${
                            selectedImageIds.includes(img.id)
                              ? "border-fuchsia-500 ring-2 ring-fuchsia-500/30"
                              : "border-zinc-700 hover:border-zinc-600"
                          }`}
                        >
                          <img
                            src={`data:${img.content_type};base64,${img.thumbnail_base64}`}
                            alt={img.filename || "Image"}
                            className="w-full h-full object-cover"
                          />
                          {selectedImageIds.includes(img.id) && (
                            <div className="absolute inset-0 bg-fuchsia-500/20 flex items-center justify-center">
                              <div className="w-6 h-6 rounded-full bg-fuchsia-500 flex items-center justify-center">
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
                              </div>
                            </div>
                          )}
                        </button>
                      ))}
                    </div>
                    {selectedImageIds.length > 0 && (
                      <button
                        onClick={() => setSelectedImageIds([])}
                        className="w-full mt-2 text-xs text-zinc-500 hover:text-zinc-400 transition-colors"
                      >
                        Clear selection
                      </button>
                    )}
                  </>
                ) : (
                  <div className="p-4 bg-fuchsia-500/10 border border-fuchsia-500/20 rounded-xl">
                    <p className="text-xs text-fuchsia-400">
                      No images uploaded. Go to Upload page to add images.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Side - Chat Interface */}
        <div className="lg:col-span-3 bg-zinc-900/60 backdrop-blur-sm rounded-2xl border border-zinc-800/80 flex flex-col min-h-[500px] max-h-[700px]">
          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
            {displayMessages.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 mb-5">
                    <Sparkles className="h-8 w-8 text-violet-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    Start a Conversation
                  </h3>
                  <p className="text-zinc-500 text-sm max-w-sm mx-auto">
                    Ask questions about your documents or select images to
                    analyze
                  </p>
                  <div className="mt-5 flex flex-wrap justify-center gap-2">
                    <span className="px-3 py-1.5 text-xs rounded-full bg-zinc-800/50 text-zinc-400 border border-zinc-700/50">
                      "What is our leave policy?"
                    </span>
                    <span className="px-3 py-1.5 text-xs rounded-full bg-zinc-800/50 text-zinc-400 border border-zinc-700/50">
                      "Describe this image"
                    </span>
                  </div>
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
          <div className="border-t border-zinc-800 p-4">
            <ChatBox onSendMessage={handleSendMessage} disabled={isStreaming} />
          </div>
        </div>
      </div>
    </div>
  );
}
