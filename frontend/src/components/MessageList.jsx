/**
 * Chat message list component displaying conversation history.
 * Modern dark theme with glassmorphism and animations.
 */
import { useState } from "react";
import { User, Bot, Sparkles, Copy, Check } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import CitationsList from "./CitationsList";

export default function MessageList({ messages, isLoading }) {
  const [copiedIndex, setCopiedIndex] = useState(null);

  const handleCopy = async (text, index) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  return (
    <div className="space-y-6">
      {messages.map((message, index) => (
        <div
          key={index}
          className="animate-slideUp"
          style={{ animationDelay: `${index * 0.03}s` }}
        >
          {/* User Message */}
          {message.type === "user" && (
            <div className="flex items-start gap-3 justify-end">
              <div className="max-w-[80%] bg-gradient-to-br from-violet-500/20 to-fuchsia-500/10 rounded-2xl rounded-tr-md px-4 py-3 border border-violet-500/10">
                <p className="text-[var(--text-secondary)] leading-relaxed text-[15px]">
                  {message.content}
                </p>
              </div>
              <div className="flex-shrink-0">
                <div className="h-8 w-8 rounded-xl bg-[var(--bg-secondary)] flex items-center justify-center border border-[var(--border-subtle)]">
                  <User className="h-4 w-4 text-[var(--text-muted)]" />
                </div>
              </div>
            </div>
          )}

          {/* Assistant Message */}
          {message.type === "assistant" && (
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0">
                <div className="relative group">
                  <div className="absolute inset-0 bg-gradient-to-br from-violet-500 to-fuchsia-500 rounded-xl blur-md opacity-40 group-hover:opacity-60 transition-opacity" />
                  <div className="relative h-8 w-8 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
                    <Bot className="h-4 w-4 text-white" />
                  </div>
                </div>
              </div>
              <div className="flex-1 space-y-3 max-w-[85%]">
                <div className="relative group">
                  <div className="bg-[var(--bg-secondary)]/60 backdrop-blur-sm rounded-2xl rounded-tl-md px-4 py-3 border border-[var(--border-subtle)] transition-colors">
                    {/* Markdown rendered content */}
                    <div
                      className="prose prose-invert prose-sm max-w-none text-[var(--text-secondary)] leading-relaxed text-[15px]
                      prose-headings:text-[var(--text-primary)] prose-headings:font-semibold prose-headings:mt-4 prose-headings:mb-2
                      prose-p:my-2 prose-p:text-[var(--text-secondary)]
                      prose-strong:text-[var(--text-primary)]
                      prose-code:text-violet-300 prose-code:bg-[var(--bg-secondary)]/60 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:text-sm prose-code:before:content-none prose-code:after:content-none
                      prose-pre:bg-[var(--bg-secondary)]/80 prose-pre:border prose-pre:border-[var(--border-subtle)] prose-pre:rounded-xl prose-pre:my-3
                      prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5
                      prose-a:text-violet-400 prose-a:no-underline hover:prose-a:text-violet-300
                      prose-blockquote:border-violet-500/50 prose-blockquote:bg-[var(--bg-secondary)]/30 prose-blockquote:rounded-r-lg prose-blockquote:my-2
                      prose-hr:border-[var(--border-subtle)]
                      prose-table:text-sm prose-th:bg-[var(--bg-secondary)]/50 prose-th:px-3 prose-th:py-2 prose-td:px-3 prose-td:py-2 prose-td:border-[var(--border-subtle)]"
                    >
                      {message.content ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {message.content}
                        </ReactMarkdown>
                      ) : message.isStreaming ? (
                        ""
                      ) : (
                        "(no content)"
                      )}
                      {/* Streaming cursor */}
                      {message.isStreaming && (
                        <span className="inline-block w-0.5 h-5 ml-0.5 bg-violet-400 rounded-full animate-pulse" />
                      )}
                    </div>

                    {/* Model info and copy button */}
                    {!message.isStreaming && (
                      <div className="mt-3 pt-3 border-t border-[var(--border-subtle)] flex items-center justify-between">
                        <div className="flex items-center">
                          {message.model && (
                            <>
                              <Sparkles className="h-3 w-3 mr-1.5 text-amber-400/80" />
                              <span className="text-xs text-[var(--text-muted)]">
                                {message.model.provider}
                                {message.model.name && (
                                  <span className="text-[var(--text-muted)] ml-1">
                                    · {message.model.name}
                                  </span>
                                )}
                              </span>
                            </>
                          )}
                        </div>
                        <button
                          onClick={() => handleCopy(message.content, index)}
                          className="p-1.5 rounded-lg hover:bg-[var(--hover-bg)] transition-colors group/copy"
                          title="Copy to clipboard"
                        >
                          {copiedIndex === index ? (
                            <Check className="h-3.5 w-3.5 text-green-400" />
                          ) : (
                            <Copy className="h-3.5 w-3.5 text-[var(--text-muted)] group-hover/copy:text-[var(--text-secondary)]" />
                          )}
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                {/* Citations */}
                {message.citations && message.citations.length > 0 && (
                  <CitationsList citations={message.citations} />
                )}
              </div>
            </div>
          )}
        </div>
      ))}

      {/* Loading indicator */}
      {isLoading &&
        (messages.length === 0 ||
          !messages[messages.length - 1]?.isStreaming) && (
          <div className="flex items-start gap-3 animate-fadeIn">
            <div className="flex-shrink-0">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-br from-violet-500 to-fuchsia-500 rounded-xl blur-md opacity-50 animate-pulse" />
                <div className="relative h-8 w-8 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
                  <Bot className="h-4 w-4 text-white" />
                </div>
              </div>
            </div>
            <div className="bg-[var(--bg-secondary)]/60 backdrop-blur-sm rounded-2xl rounded-tl-md px-4 py-3 border border-[var(--border-subtle)] transition-colors">
              <div className="flex items-center gap-2">
                <div className="flex gap-1">
                  <div
                    className="w-1.5 h-1.5 bg-violet-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0ms" }}
                  />
                  <div
                    className="w-1.5 h-1.5 bg-fuchsia-400 rounded-full animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  />
                  <div
                    className="w-1.5 h-1.5 bg-violet-400 rounded-full animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  />
                </div>
                <span className="text-[var(--text-muted)] text-sm">
                  Analyzing...
                </span>
              </div>
            </div>
          </div>
        )}
    </div>
  );
}
