// ULTIMATE System Dashboard - Real-time Monitoring & Analytics
// Complete system health, performance metrics, and advanced monitoring

import React, { useState, useEffect } from "react";
import { auth } from "../firebase";
import { apiClient } from '../config/api';

import axios from "axios";
import {
  ChartBarIcon, ServerIcon, ClockIcon, CheckCircleIcon,
  ExclamationTriangleIcon, CpuChipIcon, CircleStackIcon,
  ArrowPathIcon, EyeIcon, DocumentTextIcon, TableCellsIcon,
  PhotoIcon, ScaleIcon, BoltIcon, CloudIcon, BeakerIcon,
  RocketLaunchIcon, ShieldCheckIcon, WrenchScrewdriverIcon
} from "@heroicons/react/24/solid";

const SystemDashboard = () => {
  const [systemHealth, setSystemHealth] = useState(null);
  const [systemStats, setSystemStats] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(10); // seconds
  const [expandedSections, setExpandedSections] = useState({});
  
  const currentUser = auth.currentUser;

  // Initialize dashboard
  useEffect(() => {
    loadSystemData();
    
    if (autoRefresh) {
      const interval = setInterval(loadSystemData, refreshInterval * 1000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  // Load all system data
  const loadSystemData = async () => {
    try {
      setIsLoading(true);
      
      const [healthResponse, statsResponse] = await Promise.all([
        apiClient.get("/health"),
        apiClient.get("/system-stats")
      ]);
      
      setSystemHealth(healthResponse.data);
      setSystemStats(statsResponse.data);
      setLastUpdated(new Date());
      
    } catch (error) {
      console.error("Error loading system data:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Toggle section expansion
  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  // Format numbers with commas
  const formatNumber = (num) => {
    return new Intl.NumberFormat().format(num || 0);
  };

  // Format time duration
  const formatDuration = (ms) => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'operational':
      case 'healthy':
      case 'ready':
        return 'green';
      case 'warning':
      case 'processing':
        return 'yellow';
      case 'error':
      case 'failed':
        return 'red';
      default:
        return 'gray';
    }
  };

  // Get performance grade
  const getPerformanceGrade = (metrics) => {
    if (!metrics) return 'N/A';
    
    const successRate = metrics.success_rate_percent || 0;
    const avgTime = metrics.average_processing_times?.response_generation?.[0] || 0;
    
    if (successRate > 95 && avgTime < 2) return 'A+';
    if (successRate > 90 && avgTime < 5) return 'A';
    if (successRate > 85 && avgTime < 10) return 'B';
    if (successRate > 75 && avgTime < 15) return 'C';
    return 'D';
  };

  if (!currentUser) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <ExclamationTriangleIcon className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-800 mb-2">Authentication Required</h2>
          <p className="text-gray-600">Please sign in to access the system dashboard.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 p-6 bg-gray-50 overflow-y-auto">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-800">System Dashboard</h1>
            <p className="text-gray-600">Real-time monitoring and performance analytics</p>
          </div>
          <div className="flex items-center space-x-4">
            {/* Last Updated */}
            {lastUpdated && (
              <div className="text-sm text-gray-600">
                Updated: {lastUpdated.toLocaleTimeString()}
              </div>
            )}
            
            {/* Auto Refresh Toggle */}
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="autoRefresh"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="autoRefresh" className="text-sm text-gray-700">
                Auto-refresh ({refreshInterval}s)
              </label>
            </div>
            
            {/* Manual Refresh */}
            <button
              onClick={loadSystemData}
              disabled={isLoading}
              className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-md disabled:opacity-50"
              title="Refresh Now"
            >
              <ArrowPathIcon className={`h-5 w-5 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
        
        {/* System Status Overview */}
        {systemHealth && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gradient-to-r from-blue-500 to-blue-600 p-4 rounded-lg text-white">
              <div className="flex items-center space-x-2">
                <ServerIcon className="h-6 w-6" />
                <span className="font-medium">System Status</span>
              </div>
              <div className="text-2xl font-bold capitalize">
                {systemHealth.system_status}
              </div>
              <div className="text-sm opacity-90">
                Version {systemHealth.version}
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-green-500 to-green-600 p-4 rounded-lg text-white">
              <div className="flex items-center space-x-2">
                <CircleStackIcon className="h-6 w-6" />
                <span className="font-medium">Documents</span>
              </div>
              <div className="text-2xl font-bold">
                {formatNumber(systemHealth.uptime_info?.total_documents)}
              </div>
              <div className="text-sm opacity-90">
                Processed & Ready
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-purple-500 to-purple-600 p-4 rounded-lg text-white">
              <div className="flex items-center space-x-2">
                <BoltIcon className="h-6 w-6" />
                <span className="font-medium">Performance</span>
              </div>
              <div className="text-2xl font-bold">
                {getPerformanceGrade(systemHealth.performance_metrics)}
              </div>
              <div className="text-sm opacity-90">
                Overall Grade
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-orange-500 to-orange-600 p-4 rounded-lg text-white">
              <div className="flex items-center space-x-2">
                <RocketLaunchIcon className="h-6 w-6" />
                <span className="font-medium">Queries</span>
              </div>
              <div className="text-2xl font-bold">
                {formatNumber(systemHealth.performance_metrics?.query_statistics?.total_queries)}
              </div>
              <div className="text-sm opacity-90">
                Total Processed
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Model Status */}
      {systemHealth?.model_status && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => toggleSection('models')}
          >
            <h2 className="text-lg font-semibold text-gray-800">AI Models Status</h2>
            <EyeIcon className="h-5 w-5 text-gray-600" />
          </div>
          
          {expandedSections.models !== false && (
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {Object.entries(systemHealth.model_status).map(([model, status]) => (
                <div key={model} className="p-4 border border-gray-200 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-medium text-gray-800 capitalize">
                      {model.replace('_', ' ')}
                    </div>
                    <div className={`w-3 h-3 rounded-full ${
                      status ? 'bg-green-500' : 'bg-red-500'
                    }`}></div>
                  </div>
                  <div className="text-sm text-gray-600">
                    {status ? 'Available' : 'Unavailable'}
                  </div>
                  {model === 'gemini_flash' && status && (
                    <div className="text-xs text-blue-600 mt-1">Primary Model</div>
                  )}
                  {model === 'groq_llama' && status && (
                    <div className="text-xs text-purple-600 mt-1">Fast Inference</div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Content Statistics */}
      {systemHealth?.content_statistics && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => toggleSection('content')}
          >
            <h2 className="text-lg font-semibold text-gray-800">Content Analysis</h2>
            <ChartBarIcon className="h-5 w-5 text-gray-600" />
          </div>
          
          {expandedSections.content !== false && (
            <div className="mt-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <DocumentTextIcon className="h-8 w-8 text-blue-600 mx-auto mb-2" />
                  <div className="text-2xl font-bold text-blue-900">
                    {formatNumber(systemHealth.content_statistics.total_chunks)}
                  </div>
                  <div className="text-sm text-blue-700">Text Chunks</div>
                </div>
                
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <TableCellsIcon className="h-8 w-8 text-green-600 mx-auto mb-2" />
                  <div className="text-2xl font-bold text-green-900">
                    {formatNumber(systemHealth.content_statistics.total_tables)}
                  </div>
                  <div className="text-sm text-green-700">Tables</div>
                </div>
                
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <PhotoIcon className="h-8 w-8 text-purple-600 mx-auto mb-2" />
                  <div className="text-2xl font-bold text-purple-900">
                    {formatNumber(systemHealth.content_statistics.total_images)}
                  </div>
                  <div className="text-sm text-purple-700">Images</div>
                </div>
                
                <div className="text-center p-4 bg-orange-50 rounded-lg">
                  <ScaleIcon className="h-8 w-8 text-orange-600 mx-auto mb-2" />
                  <div className="text-2xl font-bold text-orange-900">
                    {formatNumber(systemHealth.content_statistics.total_citations)}
                  </div>
                  <div className="text-sm text-orange-700">Citations</div>
                </div>
              </div>
              
              {/* Processing Distribution Chart Placeholder */}
              <div className="bg-gray-50 p-6 rounded-lg text-center">
                <ChartBarIcon className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                <p className="text-gray-600">Content distribution visualization would go here</p>
                <p className="text-sm text-gray-500">Real-time charts require additional charting library</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Performance Metrics */}
      {systemHealth?.performance_metrics && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => toggleSection('performance')}
          >
            <h2 className="text-lg font-semibold text-gray-800">Performance Metrics</h2>
            <BoltIcon className="h-5 w-5 text-gray-600" />
          </div>
          
          {expandedSections.performance !== false && (
            <div className="mt-4">
              {/* Query Statistics */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="p-4 border border-gray-200 rounded-lg">
                  <div className="text-sm font-medium text-gray-700">Total Queries</div>
                  <div className="text-2xl font-bold text-gray-900">
                    {formatNumber(systemHealth.performance_metrics.query_statistics?.total_queries)}
                  </div>
                </div>
                
                <div className="p-4 border border-gray-200 rounded-lg">
                  <div className="text-sm font-medium text-gray-700">Success Rate</div>
                  <div className="text-2xl font-bold text-green-600">
                    {systemHealth.performance_metrics.success_rate_percent?.toFixed(1)}%
                  </div>
                </div>
                
                <div className="p-4 border border-gray-200 rounded-lg">
                  <div className="text-sm font-medium text-gray-700">Avg Response Time</div>
                  <div className="text-2xl font-bold text-blue-600">
                    {systemHealth.performance_metrics.query_statistics?.avg_response_time?.toFixed(2)}s
                  </div>
                </div>
              </div>
              
              {/* Processing Times */}
              {systemHealth.performance_metrics.average_processing_times && (
                <div className="space-y-3">
                  <h3 className="text-md font-medium text-gray-800">Average Processing Times</h3>
                  {Object.entries(systemHealth.performance_metrics.average_processing_times).map(([stage, time]) => (
                    <div key={stage} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                      <div className="capitalize text-gray-700">
                        {stage.replace('_', ' ')}
                      </div>
                      <div className="font-medium text-gray-900">
                        {formatDuration(time * 1000)}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* API Usage Statistics */}
      {systemStats && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => toggleSection('api')}
          >
            <h2 className="text-lg font-semibold text-gray-800">API Usage & System Stats</h2>
            <CloudIcon className="h-5 w-5 text-gray-600" />
          </div>
          
          {expandedSections.api !== false && (
            <div className="mt-4">
              {/* API Calls */}
              {systemHealth.performance_metrics?.api_call_counts && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  {Object.entries(systemHealth.performance_metrics.api_call_counts).map(([api, count]) => (
                    <div key={api} className="p-4 border border-gray-200 rounded-lg">
                      <div className="text-sm font-medium text-gray-700 capitalize">
                        {api.replace('_', ' ')}
                      </div>
                      <div className="text-xl font-bold text-gray-900">
                        {formatNumber(count)}
                      </div>
                      <div className="text-xs text-gray-500">API calls</div>
                    </div>
                  ))}
                </div>
              )}
              
              {/* Error Tracking */}
              {systemHealth.performance_metrics?.error_counts && (
                <div className="bg-red-50 p-4 rounded-lg">
                  <h3 className="text-md font-medium text-red-800 mb-3">Error Tracking</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {Object.entries(systemHealth.performance_metrics.error_counts).map(([type, count]) => (
                      <div key={type} className="text-center">
                        <div className="text-lg font-bold text-red-600">
                          {formatNumber(count)}
                        </div>
                        <div className="text-sm text-red-700 capitalize">
                          {type.replace('_', ' ')}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Capabilities Overview */}
      {systemHealth?.capabilities && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => toggleSection('capabilities')}
          >
            <h2 className="text-lg font-semibold text-gray-800">System Capabilities</h2>
            <ShieldCheckIcon className="h-5 w-5 text-gray-600" />
          </div>
          
          {expandedSections.capabilities !== false && (
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(systemHealth.capabilities).map(([capability, enabled]) => (
                <div key={capability} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                  <div className="flex items-center space-x-2">
                    {enabled ? (
                      <CheckCircleIcon className="h-5 w-5 text-green-500" />
                    ) : (
                      <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
                    )}
                    <span className="text-gray-700 capitalize">
                      {capability.replace(/_/g, ' ')}
                    </span>
                  </div>
                  <span className={`text-sm font-medium ${
                    enabled ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {enabled ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Refresh Interval Control */}
      <div className="mt-6 bg-white rounded-lg shadow-md p-4">
        <div className="flex items-center justify-between">
          <div className="text-sm font-medium text-gray-700">Auto-refresh interval</div>
          <div className="flex items-center space-x-2">
            <input
              type="range"
              min="5"
              max="60"
              step="5"
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(parseInt(e.target.value))}
              className="w-24 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              disabled={!autoRefresh}
            />
            <span className="text-sm text-gray-600 w-8">{refreshInterval}s</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemDashboard;