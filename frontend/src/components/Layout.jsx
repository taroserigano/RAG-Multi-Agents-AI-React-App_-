/**
 * Layout component with navigation and common structure.
 */
import { Link, useLocation } from "react-router-dom";
import { FileText, MessageSquare } from "lucide-react";

export default function Layout({ children }) {
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top Navigation */}
      <nav className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo/Title */}
            <div className="flex items-center">
              <FileText className="h-8 w-8 text-primary-600" />
              <h1 className="ml-2 text-xl font-bold text-gray-900">
                Policy RAG
              </h1>
            </div>

            {/* Navigation Links */}
            <div className="flex space-x-4">
              <Link
                to="/upload"
                className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive("/upload")
                    ? "bg-primary-100 text-primary-700"
                    : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                }`}
              >
                <FileText className="h-4 w-4 mr-2" />
                Upload
              </Link>

              <Link
                to="/chat"
                className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive("/chat")
                    ? "bg-primary-100 text-primary-700"
                    : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                }`}
              >
                <MessageSquare className="h-4 w-4 mr-2" />
                Chat
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
    </div>
  );
}
