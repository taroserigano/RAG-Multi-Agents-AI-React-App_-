/**
 * Compliance checking page.
 * Combines document and image analysis for thorough compliance assessment.
 */
import { useState } from "react";
import { AlertCircle } from "lucide-react";
import ComplianceChecker from "../components/ComplianceChecker";
import ModelPicker from "../components/ModelPicker";
import { useDocuments, useImages } from "../hooks/useApi";

// Generate a simple session ID for user tracking
const getUserId = () => {
  let userId = localStorage.getItem("user_id");
  if (!userId) {
    userId = `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem("user_id", userId);
  }
  return userId;
};

export default function CompliancePage() {
  const [selectedProvider, setSelectedProvider] = useState("openai");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedDocIds, setSelectedDocIds] = useState([]);
  const [selectedImageIds, setSelectedImageIds] = useState([]);

  const userId = getUserId();

  // Fetch documents
  const {
    data: documents = [],
    isLoading: docsLoading,
    error: docsError,
  } = useDocuments();

  // Fetch images
  const {
    data: images = [],
    isLoading: imagesLoading,
    error: imagesError,
  } = useImages();

  const handleProviderChange = (newProvider) => {
    setSelectedProvider(newProvider);
    setSelectedModel("");
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header with model picker */}
      <div className="bg-[var(--bg-secondary)] border-b border-[var(--border-subtle)] px-4 py-3 flex items-center justify-between rounded-t-xl transition-colors duration-300">
        <div>
          <h1 className="text-lg font-semibold text-[var(--text-primary)]">
            Compliance Checker
          </h1>
          <p className="text-sm text-[var(--text-secondary)]">
            Analyze documents and images for policy compliance
          </p>
        </div>
        <div className="flex items-center gap-4">
          <ModelPicker
            selectedProvider={selectedProvider}
            selectedModel={selectedModel}
            onProviderChange={handleProviderChange}
            onModelChange={setSelectedModel}
          />
        </div>
      </div>

      {/* Error display */}
      {(docsError || imagesError) && (
        <div className="m-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2 text-red-400">
          <AlertCircle className="w-4 h-4" />
          <span className="text-sm">
            {docsError && `Documents: ${docsError.message}`}
            {docsError && imagesError && " | "}
            {imagesError && `Images: ${imagesError.message}`}
          </span>
        </div>
      )}

      {/* Loading state */}
      {(docsLoading || imagesLoading) && (
        <div className="m-4 p-3 bg-violet-500/10 border border-violet-500/20 rounded-lg">
          <span className="text-sm text-violet-400">Loading resources...</span>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 overflow-hidden">
        <ComplianceChecker
          documents={documents}
          images={images}
          selectedDocIds={selectedDocIds}
          selectedImageIds={selectedImageIds}
          provider={selectedProvider}
          model={selectedModel}
          userId={userId}
          onSelectDocuments={setSelectedDocIds}
          onSelectImages={setSelectedImageIds}
        />
      </div>
    </div>
  );
}
