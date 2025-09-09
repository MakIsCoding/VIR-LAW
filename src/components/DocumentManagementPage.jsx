// FIXED DocumentManagementPage.jsx - All advanced features preserved with fixes
import React, { useState, useEffect, useCallback } from "react";
import { auth } from "../firebase";
import axios from "axios";
import {
  DocumentPlusIcon, DocumentIcon, TrashIcon, ArrowUpTrayIcon,
  CheckCircleIcon, ExclamationTriangleIcon, ClockIcon,
  ChartBarIcon, CogIcon, ArrowPathIcon, InformationCircleIcon,
  DocumentTextIcon, TableCellsIcon, PhotoIcon, ScaleIcon,
  FolderOpenIcon, CloudArrowUpIcon, ServerIcon
} from "@heroicons/react/24/solid";

const DocumentManagementPage = () => {
  const [documents, setDocuments] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [uploadProgress, setUploadProgress] = useState({});
  const [processingStatus, setProcessingStatus] = useState({});
  const [systemHealth, setSystemHealth] = useState(null);
  const [processingStats, setProcessingStats] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [batchSize, setBatchSize] = useState(5);
  const [autoProcess, setAutoProcess] = useState(true);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);

  const currentUser = auth.currentUser;

  // Initialize component
  useEffect(() => {
    loadDocuments();
    loadSystemHealth();
    loadProcessingStats();

    // Set up polling for real-time updates
    const interval = setInterval(() => {
      loadSystemHealth();
      if (isProcessing) {
        loadProcessingStats();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [isProcessing]);

  // Load document list
  const loadDocuments = async () => {
    try {
      setIsLoading(true);
      const response = await axios.get("http://localhost:8000/document-list");
      setDocuments(response.data.documents || []);
    } catch (error) {
      console.error("Error loading documents:", error);
      setUploadError("Failed to load documents. Please check backend connection.");
    } finally {
      setIsLoading(false);
    }
  };

  // Load system health
  const loadSystemHealth = async () => {
    try {
      const response = await axios.get("http://localhost:8000/health");
      setSystemHealth(response.data);
    } catch (error) {
      console.error("Error loading system health:", error);
    }
  };

  // Load processing statistics
  const loadProcessingStats = async () => {
    try {
      const response = await axios.get("http://localhost:8000/system-stats");
      setProcessingStats(response.data);
    } catch (error) {
      console.error("Error loading processing stats:", error);
    }
  };

  // Handle file selection
  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
    setUploadError(null);
    setUploadSuccess(null);
  };

  // Handle drag and drop
  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const files = Array.from(e.dataTransfer.files);
    setSelectedFiles(files);
    setUploadError(null);
    setUploadSuccess(null);
  }, []);

  // Upload files
  const uploadFiles = async () => {
    if (selectedFiles.length === 0) return;

    try {
      setUploadError(null);
      setUploadSuccess(null);
      
      const formData = new FormData();
      selectedFiles.forEach(file => {
        formData.append('files', file);
      });

      // Track upload progress
      selectedFiles.forEach(file => {
        setUploadProgress(prev => ({
          ...prev,
          [file.name]: { status: 'uploading', progress: 0 }
        }));
      });

      const response = await axios.post("http://localhost:8000/upload-documents", formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          selectedFiles.forEach(file => {
            setUploadProgress(prev => ({
              ...prev,
              [file.name]: { status: 'uploading', progress: percentCompleted }
            }));
          });
        }
      });

      const { uploaded, failed } = response.data;

      // Update progress status
      uploaded.forEach(file => {
        setUploadProgress(prev => ({
          ...prev,
          [file.filename]: { status: 'success', progress: 100 }
        }));
      });

      failed.forEach(file => {
        setUploadProgress(prev => ({
          ...prev,
          [file.filename]: { status: 'error', progress: 0, error: file.error }
        }));
      });

      setUploadSuccess(`Successfully uploaded ${uploaded.length} file(s).`);
      if (failed.length > 0) {
        setUploadError(`Failed to upload ${failed.length} file(s).`);
      }

      // Clear selection and reload documents
      setSelectedFiles([]);
      await loadDocuments();

      // Auto-process if enabled
      if (autoProcess && uploaded.length > 0) {
        await processDocuments();
      }

    } catch (error) {
      console.error("Error uploading files:", error);
      setUploadError("Failed to upload files. Please try again.");
      
      // Update all files as failed
      selectedFiles.forEach(file => {
        setUploadProgress(prev => ({
          ...prev,
          [file.name]: { status: 'error', progress: 0, error: error.message }
        }));
      });
    }
  };

  // Process documents
  const processDocuments = async () => {
    try {
      setIsProcessing(true);
      setUploadError(null);
      setUploadSuccess(null);

      // FIXED: Use correct endpoint
      const response = await axios.post("http://localhost:8000/process-documents", {
        documents_dir: "./legal_documents",
        batch_size: batchSize
      });

      if (response.data.status === "success" || response.data.success) {
        setUploadSuccess("Documents processed successfully!");
        setProcessingStats(response.data.statistics || response.data.detailed_results);
        await loadDocuments();
        await loadSystemHealth();
      } else {
        setUploadError(`Processing failed: ${response.data.message}`);
      }

    } catch (error) {
      console.error("Error processing documents:", error);
      setUploadError("Failed to process documents. Please check backend connection.");
    } finally {
      setIsProcessing(false);
    }
  };

  // Delete document
  const deleteDocument = async (filename) => {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) return;

    try {
      // This would need a backend endpoint implementation
      console.log("Delete functionality needs backend endpoint");
      setUploadError("Delete functionality requires backend implementation.");
    } catch (error) {
      console.error("Error deleting document:", error);
      setUploadError("Failed to delete document.");
    }
  };

  // Get file icon based on type
  const getFileIcon = (filename) => {
    const ext = filename.toLowerCase().split('.').pop();
    const iconClass = "h-8 w-8";

    switch (ext) {
      case 'pdf':
        return <DocumentIcon className={`${iconClass} text-red-500`} />;
      case 'docx':
      case 'doc':
        return <DocumentTextIcon className={`${iconClass} text-blue-500`} />;
      case 'txt':
        return <DocumentTextIcon className={`${iconClass} text-gray-500`} />;
      case 'xlsx':
      case 'xls':
        return <TableCellsIcon className={`${iconClass} text-green-500`} />;
      case 'png':
      case 'jpg':
      case 'jpeg':
      case 'gif':
        return <PhotoIcon className={`${iconClass} text-purple-500`} />;
      default:
        return <DocumentIcon className={`${iconClass} text-gray-400`} />;
    }
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (!currentUser) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <ExclamationTriangleIcon className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-800 mb-2">Authentication Required</h2>
          <p className="text-gray-600">Please sign in to access document management.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 p-6 bg-gray-50">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <DocumentPlusIcon className="h-8 w-8 text-blue-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-800">Document Management</h1>
              <p className="text-gray-600">Upload and process legal documents for AI analysis</p>
            </div>
          </div>
          
          {/* System Status Indicator */}
          {systemHealth && (
            <div className="flex items-center space-x-2 text-sm">
              {systemHealth.status === "healthy" ? (
                <CheckCircleIcon className="h-5 w-5 text-green-500" />
              ) : (
                <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
              )}
              <span className={systemHealth.status === "healthy" ? "text-green-600" : "text-red-600"}>
                {systemHealth.status === "healthy" ? "Backend Online" : "Backend Offline"}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Advanced Options Toggle */}
      <div className="bg-white rounded-lg shadow-md p-4 mb-6">
        <button
          onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
          className="flex items-center space-x-2 text-gray-700 hover:text-gray-900"
        >
          <CogIcon className="h-5 w-5" />
          <span>Advanced Options</span>
          <ArrowPathIcon className={`h-4 w-4 transform transition-transform ${showAdvancedOptions ? 'rotate-180' : ''}`} />
        </button>

        {showAdvancedOptions && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Batch Size
              </label>
              <input
                type="number"
                min="1"
                max="10"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            
            <div className="flex items-center">
              <input
                type="checkbox"
                id="autoProcess"
                checked={autoProcess}
                onChange={(e) => setAutoProcess(e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="autoProcess" className="ml-2 block text-sm text-gray-700">
                Auto-process after upload
              </label>
            </div>

            <button
              onClick={processDocuments}
              disabled={isProcessing}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
            >
              {isProcessing ? "Processing..." : "Process Documents"}
            </button>
          </div>
        )}
      </div>

      {/* Upload Area */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragActive 
              ? "border-blue-500 bg-blue-50" 
              : "border-gray-300 hover:border-gray-400"
          }`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <CloudArrowUpIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-800 mb-2">
            Drag and drop files here, or click to select
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            Supported formats: PDF, DOCX, DOC, TXT, RTF, PNG, JPG, JPEG, TIFF
          </p>
          
          <input
            type="file"
            multiple
            onChange={handleFileSelect}
            className="hidden"
            id="file-upload"
            accept=".pdf,.docx,.doc,.txt,.rtf,.png,.jpg,.jpeg,.tiff"
          />
          <label
            htmlFor="file-upload"
            className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 cursor-pointer"
          >
            <ArrowUpTrayIcon className="h-5 w-5 mr-2" />
            Select Files
          </label>
        </div>

        {/* Selected Files Display */}
        {selectedFiles.length > 0 && (
          <div className="mt-6">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Selected Files:</h4>
            <div className="space-y-2">
              {selectedFiles.map((file, index) => {
                const progress = uploadProgress[file.name];
                return (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      {getFileIcon(file.name)}
                      <div>
                        <p className="text-sm font-medium text-gray-900">{file.name}</p>
                        <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                      </div>
                    </div>
                    
                    {progress && (
                      <div className="flex items-center space-x-2">
                        {progress.status === 'uploading' && (
                          <>
                            <div className="w-16 bg-gray-200 rounded-full h-2">
                              <div 
                                className="bg-blue-600 h-2 rounded-full transition-all"
                                style={{ width: `${progress.progress}%` }}
                              />
                            </div>
                            <span className="text-xs text-gray-500">{progress.progress}%</span>
                          </>
                        )}
                        {progress.status === 'success' && (
                          <CheckCircleIcon className="h-5 w-5 text-green-500" />
                        )}
                        {progress.status === 'error' && (
                          <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            
            <button
              onClick={uploadFiles}
              disabled={selectedFiles.length === 0}
              className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              Upload Files
            </button>
          </div>
        )}
      </div>

      {/* Status Messages */}
      {(uploadError || uploadSuccess) && (
        <div className="bg-white rounded-lg shadow-md p-4 mb-6">
          {uploadSuccess && (
            <div className="flex items-center space-x-2 text-green-700 mb-2">
              <CheckCircleIcon className="h-5 w-5" />
              <span>{uploadSuccess}</span>
            </div>
          )}
          {uploadError && (
            <div className="flex items-center space-x-2 text-red-700">
              <ExclamationTriangleIcon className="h-5 w-5" />
              <span>{uploadError}</span>
            </div>
          )}
        </div>
      )}

      {/* Document List */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800">Uploaded Documents</h3>
          <button
            onClick={loadDocuments}
            className="px-3 py-1 text-sm bg-gray-100 text-gray-600 rounded hover:bg-gray-200"
          >
            <ArrowPathIcon className="h-4 w-4 inline mr-1" />
            Refresh
          </button>
        </div>

        {isLoading ? (
          <div className="text-center py-8">
            <ClockIcon className="h-8 w-8 text-gray-400 mx-auto mb-2 animate-spin" />
            <p className="text-gray-500">Loading documents...</p>
          </div>
        ) : documents.length === 0 ? (
          <div className="text-center py-8">
            <FolderOpenIcon className="h-12 w-12 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-500 mb-2">No documents uploaded yet.</p>
            <p className="text-sm text-gray-400">Upload legal documents to get started.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {documents.map((doc, index) => (
              <div key={index} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg hover:bg-gray-50">
                <div className="flex items-center space-x-3">
                  {getFileIcon(doc)}
                  <div>
                    <p className="text-sm font-medium text-gray-900">{doc}</p>
                    <p className="text-xs text-gray-500">Legal Document</p>
                  </div>
                </div>
                
                <button
                  onClick={() => deleteDocument(doc)}
                  className="p-2 text-red-500 hover:bg-red-50 rounded"
                >
                  <TrashIcon className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Processing Stats */}
      {processingStats && (
        <div className="mt-6 bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Processing Statistics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-blue-600">{processingStats.processing_stats?.total_documents || 0}</p>
              <p className="text-sm text-gray-500">Documents</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-green-600">{processingStats.processing_stats?.total_chunks || 0}</p>
              <p className="text-sm text-gray-500">Chunks</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-purple-600">{processingStats.processing_stats?.total_tables || 0}</p>
              <p className="text-sm text-gray-500">Tables</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-orange-600">{processingStats.processing_stats?.total_images || 0}</p>
              <p className="text-sm text-gray-500">Images</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentManagementPage;