// ENHANCED DocumentManagementPage.jsx - Constitutional intelligence for processing only
import React, { useState, useEffect, useCallback } from "react";
import { auth } from "../firebase";
import { apiClient, API_BASE_URL, HF_TOKEN } from '../config/api';
import axios from "axios";
import {
  DocumentPlusIcon, DocumentIcon, TrashIcon, ArrowUpTrayIcon,
  CheckCircleIcon, ExclamationTriangleIcon, ClockIcon,
  ChartBarIcon, CogIcon, ArrowPathIcon, InformationCircleIcon,
  DocumentTextIcon, TableCellsIcon, PhotoIcon, ScaleIcon,
  FolderOpenIcon, CloudArrowUpIcon, ServerIcon, 
  BookOpenIcon, AcademicCapIcon, ShieldCheckIcon
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

  // Enhanced: Load document list with constitutional analysis
  const loadDocuments = async () => {
    try {
      setIsLoading(true);
      const response = await apiClient.get("/document-list");
      setDocuments(response.data.documents || []);
    } catch (error) {
      console.error("Error loading documents:", error);
      setUploadError("Failed to load documents. Please check backend connection.");
    } finally {
      setIsLoading(false);
    }
  };

  // Enhanced: Load system health with constitutional readiness
  const loadSystemHealth = async () => {
    try {
      const response = await apiClient.get("/health");
      setSystemHealth(response.data);
    } catch (error) {
      console.error("Error loading system health:", error);
    }
  };

  // Enhanced: Load processing statistics with legal intelligence
  const loadProcessingStats = async () => {
    try {
      const response = await apiClient.get("/system-stats");
      setProcessingStats(response.data);
    } catch (error) {
      console.error("Error loading processing stats:", error);
    }
  };

  // Your existing file handling functions preserved
  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
    setUploadError(null);
    setUploadSuccess(null);
  };

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

  // Enhanced: Upload files with constitutional document detection
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

      const response = await axios.post(`${API_BASE_URL}/upload-documents`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          ...(HF_TOKEN && { 'Authorization': `Bearer ${HF_TOKEN}` })
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

      const { uploaded, failed, constitutional_documents, upload_summary } = response.data;

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

      // Enhanced: Constitutional document success message
      if (constitutional_documents && constitutional_documents.length > 0) {
        setUploadSuccess(
          `Successfully uploaded ${uploaded.length} file(s) including ${constitutional_documents.length} constitutional document(s): ${constitutional_documents.join(', ')}`
        );
      } else {
        setUploadSuccess(`Successfully uploaded ${uploaded.length} file(s).`);
      }
      
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

  // Enhanced: Process documents with constitutional intelligence
  const processDocuments = async () => {
    try {
      setIsProcessing(true);
      setUploadError(null);
      setUploadSuccess(null);

      // Enhanced: Use ultimate processing endpoint with constitutional intelligence
      const response = await apiClient.post("/process-documents-ultimate", {
        documents_dir: "./legal_documents",
        batch_size: batchSize
      });

      if (response.data.status === "success" || response.data.success) {
        const stats = response.data.statistics || response.data.detailed_results;
        
        // Enhanced: Constitutional processing success message
        const constitutionalDocs = stats?.legal_analysis?.constitutional_documents || 0;
        const totalArticles = stats?.legal_analysis?.total_articles_processed || 0;
        
        if (constitutionalDocs > 0) {
          setUploadSuccess(
            `Documents processed successfully! Processed ${constitutionalDocs} constitutional document(s) with ${totalArticles} articles. System ready for constitutional law queries.`
          );
        } else {
          setUploadSuccess("Documents processed successfully!");
        }
        
        setProcessingStats(stats);
        await loadDocuments();
        await loadSystemHealth();
      } else {
        setUploadError(`Processing failed: ${response.data.message || 'Unknown error'}`);
      }

    } catch (error) {
      console.error("Error processing documents:", error);
      setUploadError("Failed to process documents. Please check backend connection.");
    } finally {
      setIsProcessing(false);
    }
  };

  // Your existing delete function preserved
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

  // Enhanced: Get file icon with constitutional document detection
  const getFileIcon = (filename, isConstitutional = false) => {
    const ext = filename.toLowerCase().split('.').pop();
    const iconClass = "h-8 w-8";

    // Enhanced: Special icon for constitutional documents
    if (isConstitutional || filename.toLowerCase().includes('constitution')) {
      return <BookOpenIcon className={`${iconClass} text-amber-600`} />;
    }

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

  // Your existing formatFileSize function preserved
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Enhanced: Determine document type for display
  const getDocumentType = (doc) => {
    const filename = typeof doc === 'string' ? doc : doc.filename;
    const isConstitutional = filename.toLowerCase().includes('constitution') || 
                           filename.toLowerCase().includes('fundamental') ||
                           filename.toLowerCase().includes('rights');
    
    if (isConstitutional) return { type: 'Constitutional Document', priority: 'high', color: 'text-amber-600' };
    if (filename.toLowerCase().includes('case') || filename.toLowerCase().includes('judgment')) {
      return { type: 'Case Law Document', priority: 'medium', color: 'text-blue-600' };
    }
    return { type: 'Legal Document', priority: 'normal', color: 'text-gray-500' };
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
      {/* Enhanced Header with Constitutional Intelligence */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <DocumentPlusIcon className="h-8 w-8 text-blue-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-800">Document Management</h1>
              <p className="text-gray-600">Upload and process legal documents for constitutional AI analysis</p>
            </div>
          </div>
          
          {/* Enhanced System Status with Constitutional Readiness */}
          {systemHealth && (
            <div className="flex items-center space-x-4">
              {/* Constitutional Readiness Indicator */}
              {systemHealth.legal_system_readiness?.constitution_processed && (
                <div className="flex items-center space-x-2 text-sm">
                  <ShieldCheckIcon className="h-5 w-5 text-amber-500" />
                  <span className="text-amber-600">Constitutional Ready</span>
                </div>
              )}
              
              {/* Backend Status */}
              <div className="flex items-center space-x-2 text-sm">
                {systemHealth.system_status === "operational" ? (
                  <CheckCircleIcon className="h-5 w-5 text-green-500" />
                ) : (
                  <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
                )}
                <span className={systemHealth.system_status === "operational" ? "text-green-600" : "text-red-600"}>
                  {systemHealth.system_status === "operational" ? "Backend Online" : "Backend Offline"}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Enhanced Advanced Options with Constitutional Settings */}
      <div className="bg-white rounded-lg shadow-md p-4 mb-6">
        <button
          onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
          className="flex items-center space-x-2 text-gray-700 hover:text-gray-900"
        >
          <CogIcon className="h-5 w-5" />
          <span>Advanced Processing Options</span>
          <ArrowPathIcon className={`h-4 w-4 transform transition-transform ${showAdvancedOptions ? 'rotate-180' : ''}`} />
        </button>

        {showAdvancedOptions && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4">
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

            {/* Enhanced: Constitutional Processing Priority */}
            <div className="flex items-center">
              <div className="flex items-center space-x-2">
                <BookOpenIcon className="h-4 w-4 text-amber-500" />
                <span className="text-sm text-gray-700">Constitutional Priority</span>
                <InformationCircleIcon className="h-4 w-4 text-gray-400" title="Constitutional documents are processed first" />
              </div>
            </div>

            <button
              onClick={processDocuments}
              disabled={isProcessing}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 flex items-center space-x-2"
            >
              {isProcessing ? (
                <>
                  <ClockIcon className="h-4 w-4 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <CogIcon className="h-4 w-4" />
                  <span>Process Documents</span>
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* Enhanced Upload Area */}
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
            Drag and drop legal documents here, or click to select
          </h3>
          <p className="text-sm text-gray-600 mb-2">
            Supported formats: PDF, DOCX, DOC, TXT, RTF, PNG, JPG, JPEG, TIFF
          </p>
          <p className="text-xs text-amber-600 mb-4">
            💡 Constitutional documents (Constitution, Fundamental Rights) are automatically prioritized
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
            Select Legal Documents
          </label>
        </div>

        {/* Enhanced Selected Files Display with Constitutional Detection */}
        {selectedFiles.length > 0 && (
          <div className="mt-6">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Selected Files:</h4>
            <div className="space-y-2">
              {selectedFiles.map((file, index) => {
                const progress = uploadProgress[file.name];
                const isConstitutional = file.name.toLowerCase().includes('constitution') || 
                                       file.name.toLowerCase().includes('fundamental');
                return (
                  <div key={index} className={`flex items-center justify-between p-3 rounded-lg ${
                    isConstitutional ? 'bg-amber-50 border border-amber-200' : 'bg-gray-50'
                  }`}>
                    <div className="flex items-center space-x-3">
                      {getFileIcon(file.name, isConstitutional)}
                      <div>
                        <p className="text-sm font-medium text-gray-900">{file.name}</p>
                        <div className="flex items-center space-x-2">
                          <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                          {isConstitutional && (
                            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
                              Constitutional
                            </span>
                          )}
                        </div>
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

      {/* Your existing Status Messages section preserved */}
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

      {/* Enhanced Document List with Constitutional Classification */}
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
            <p className="text-sm text-gray-400">Upload legal documents to get started with constitutional analysis.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {documents.map((doc, index) => {
              const docInfo = getDocumentType(doc);
              const filename = typeof doc === 'string' ? doc : doc.filename;
              const isConstitutional = docInfo.type === 'Constitutional Document';
              
              return (
                <div key={index} className={`flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50 ${
                  isConstitutional ? 'border-amber-200 bg-amber-50' : 'border-gray-200'
                }`}>
                  <div className="flex items-center space-x-3">
                    {getFileIcon(filename, isConstitutional)}
                    <div>
                      <p className="text-sm font-medium text-gray-900">{filename}</p>
                      <div className="flex items-center space-x-2">
                        <p className={`text-xs ${docInfo.color}`}>{docInfo.type}</p>
                        {docInfo.priority === 'high' && (
                          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
                            High Priority
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <button
                    onClick={() => deleteDocument(filename)}
                    className="p-2 text-red-500 hover:bg-red-50 rounded"
                  >
                    <TrashIcon className="h-4 w-4" />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Enhanced Processing Stats with Constitutional Intelligence */}
      {processingStats && (
        <div className="mt-6 bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Processing Statistics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
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

          {/* Enhanced: Constitutional Analysis Stats */}
          {processingStats.legal_intelligence && (
            <div className="border-t pt-4">
              <h4 className="text-md font-medium text-gray-700 mb-3">Constitutional Analysis</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <p className="text-xl font-bold text-amber-600">
                    {processingStats.legal_intelligence.constitutional_documents_available ? '✓' : '✗'}
                  </p>
                  <p className="text-xs text-gray-500">Constitution Available</p>
                </div>
                <div className="text-center">
                  <p className="text-xl font-bold text-amber-600">
                    {processingStats.legal_intelligence.total_constitutional_articles || 0}
                  </p>
                  <p className="text-xs text-gray-500">Articles Processed</p>
                </div>
                <div className="text-center">
                  <p className="text-xl font-bold text-amber-600">
                    {processingStats.legal_intelligence.constitutional_parts_coverage || 0}
                  </p>
                  <p className="text-xs text-gray-500">Constitutional Parts</p>
                </div>
                <div className="text-center">
                  <p className="text-xl font-bold text-amber-600">
                    {processingStats.legal_intelligence.legal_citation_database_size || 0}
                  </p>
                  <p className="text-xs text-gray-500">Legal Citations</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DocumentManagementPage;
