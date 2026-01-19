/**
 * Chat input box component for sending messages.
 * Modern dark theme with gradient effects.
 */
import { useState } from "react";
import { Send, Sparkles } from "lucide-react";

export default function ChatBox({ onSendMessage, disabled = false }) {
  const [message, setMessage] = useState("");
  const [isFocused, setIsFocused] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();

    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage("");
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative">
      {/* Glow effect when focused */}
      <div
        className={`absolute -inset-0.5 bg-gradient-to-r from-violet-500 via-fuchsia-500 to-violet-500 rounded-2xl blur-md transition-opacity duration-300 ${isFocused ? "opacity-25" : "opacity-0"}`}
      />

      <div className="relative bg-[var(--bg-secondary)]/80 backdrop-blur-xl rounded-2xl border border-[var(--border-subtle)] overflow-hidden transition-colors">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder="Ask a question about your documents..."
          disabled={disabled}
          rows={3}
          className="w-full px-4 py-4 pr-14 bg-transparent text-[var(--text-secondary)] placeholder-[var(--text-muted)] focus:outline-none resize-none disabled:opacity-50 disabled:cursor-not-allowed text-[15px]"
        />

        <button
          type="submit"
          disabled={disabled || !message.trim()}
          className={`absolute right-3 bottom-3 p-2.5 rounded-xl transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed group ${
            message.trim() && !disabled
              ? "bg-gradient-to-br from-violet-500 to-fuchsia-500 shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40 hover:scale-105"
              : "bg-[var(--bg-secondary)]"
          }`}
        >
          <Send
            className={`h-4 w-4 transition-transform duration-200 ${
              message.trim() && !disabled
                ? "text-white group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
                : "text-[var(--text-muted)]"
            }`}
          />
        </button>
      </div>

      <div className="mt-2.5 flex items-center justify-between text-xs text-[var(--text-muted)]">
        <div className="flex items-center gap-1.5">
          <Sparkles className="h-3 w-3 text-amber-500/70" />
          <span>AI-powered document analysis</span>
        </div>
        <span className="text-[var(--text-muted)]">
          Enter ↵ send · Shift+Enter new line
        </span>
      </div>
    </form>
  );
}
