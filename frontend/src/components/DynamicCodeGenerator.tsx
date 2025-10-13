import React, { useState, useEffect } from 'react';
import { CodeGenerationRequest, CodeGenerationResponse } from '../types/api';

interface DynamicCodeGeneratorProps {
  onCodeGenerated?: (response: CodeGenerationResponse) => void;
}

interface Peripheral {
  name: string;
  operations: string[];
  descriptions: Record<string, string>;
  required_parameters: Record<string, string[]>;
}

interface PerformanceMetrics {
  total_requests: number;
  avg_latency: number;
  p50_latency: number;
  p95_latency: number;
  avg_throughput: number;
  total_tokens: number;
}

const DynamicCodeGenerator: React.FC<DynamicCodeGeneratorProps> = ({ onCodeGenerated }) => {
  const [microcontrollers, setMicrocontrollers] = useState<string[]>([]);
  const [peripherals, setPeripherals] = useState<string[]>([]);
  const [selectedPeripheral, setSelectedPeripheral] = useState<string>('');
  const [peripheralData, setPeripheralData] = useState<Peripheral | null>(null);
  const [operations, setOperations] = useState<string[]>([]);
  const [selectedOperation, setSelectedOperation] = useState<string>('');
  const [languages, setLanguages] = useState<string[]>([]);
  const [selectedLanguage, setSelectedLanguage] = useState<string>('c');
  const [includeComments, setIncludeComments] = useState<boolean>(true);
  const [includeErrorHandling, setIncludeErrorHandling] = useState<boolean>(true);
  const [parameters, setParameters] = useState<Record<string, string>>({});
  const [generatedCode, setGeneratedCode] = useState<string>('');
  const [explanation, setExplanation] = useState<string>('');
  const [dependencies, setDependencies] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(null);
  const [modelInfo, setModelInfo] = useState<any>(null);

  // Fetch data from backend on component mount
  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    try {
      setLoading(true);
      
      // Fetch microcontrollers
      try {
        const mcuResponse = await fetch('http://localhost:8000/microcontrollers');
        if (mcuResponse.ok) {
          const mcuData = await mcuResponse.json();
          setMicrocontrollers(Array.isArray(mcuData) ? mcuData : []);
        } else {
          setMicrocontrollers([]);
        }
      } catch (err) {
        console.warn('Failed to fetch microcontrollers:', err);
        setMicrocontrollers([]);
      }
      
      // Fetch peripherals
      try {
        const peripheralResponse = await fetch('http://localhost:8000/peripherals');
        if (peripheralResponse.ok) {
          const peripheralData = await peripheralResponse.json();
          setPeripherals(Array.isArray(peripheralData) ? peripheralData : []);
        } else {
          setPeripherals([]);
        }
      } catch (err) {
        console.warn('Failed to fetch peripherals:', err);
        setPeripherals([]);
      }
      
      // Fetch languages
      try {
        const languageResponse = await fetch('http://localhost:8000/languages');
        if (languageResponse.ok) {
          const languageData = await languageResponse.json();
          setLanguages(Array.isArray(languageData) ? languageData : ['c', 'cpp']);
        } else {
          setLanguages(['c', 'cpp']);
        }
      } catch (err) {
        console.warn('Failed to fetch languages:', err);
        setLanguages(['c', 'cpp']);
      }
      
      // Fetch performance metrics
      try {
        const performanceResponse = await fetch('http://localhost:8000/performance');
        if (performanceResponse.ok) {
          const performanceData = await performanceResponse.json();
          setPerformanceMetrics(performanceData.performance_metrics);
        }
      } catch (err) {
        console.warn('Failed to fetch performance metrics:', err);
      }
      
      // Fetch model info
      try {
        const modelResponse = await fetch('http://localhost:8000/model/info');
        if (modelResponse.ok) {
          const modelData = await modelResponse.json();
          setModelInfo(modelData);
        }
      } catch (err) {
        console.warn('Failed to fetch model info:', err);
      }
      
    } catch (err) {
      setError('Failed to fetch initial data from backend');
      console.error('Error fetching initial data:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchPeripheralOperations = async (peripheral: string) => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:8000/operations/${peripheral}`);
      const data = await response.json();
      setPeripheralData(data);
      setOperations(data.operations);
      setSelectedOperation('');
      setParameters({});
    } catch (err) {
      setError(`Failed to fetch operations for ${peripheral}`);
      console.error('Error fetching peripheral operations:', err);
    } finally {
      setLoading(false);
    }
  };

  const handlePeripheralChange = (peripheral: string) => {
    setSelectedPeripheral(peripheral);
    if (peripheral) {
      fetchPeripheralOperations(peripheral);
    } else {
      setPeripheralData(null);
      setOperations([]);
      setSelectedOperation('');
      setParameters({});
    }
  };

  const handleOperationChange = (operation: string) => {
    setSelectedOperation(operation);
    if (operation && peripheralData) {
      const requiredParams = peripheralData.required_parameters[operation] || [];
      const newParameters: Record<string, string> = {};
      requiredParams.forEach(param => {
        newParameters[param] = '';
      });
      setParameters(newParameters);
    } else {
      setParameters({});
    }
  };

  const handleParameterChange = (paramName: string, value: string) => {
    setParameters(prev => ({
      ...prev,
      [paramName]: value
    }));
  };

  const generateCode = async () => {
    if (!selectedPeripheral || !selectedOperation || !microcontrollers.length) {
      setError('Please select a peripheral, operation, and microcontroller');
      return;
    }

    try {
      setLoading(true);
      setError('');

      const request: CodeGenerationRequest = {
        peripheral: selectedPeripheral,
        microcontroller: microcontrollers[0] || 'arduino', // Use first available microcontroller or default
        operation: selectedOperation,
        parameters: parameters,
        language: selectedLanguage,
        include_comments: includeComments,
        include_error_handling: includeErrorHandling,
      };

      const response = await fetch('http://localhost:8000/generate-code', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail?.error || 'Failed to generate code');
      }

      const result: CodeGenerationResponse = await response.json();
      
      setGeneratedCode(result.code);
      setExplanation(result.explanation);
      setDependencies(result.dependencies || []);
      setWarnings(result.warnings || []);

      if (onCodeGenerated) {
        onCodeGenerated(result);
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate code');
      console.error('Error generating code:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSimpleQuery = async (query: string) => {
    try {
      setLoading(true);
      setError('');

      const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate response');
      }

      const result = await response.json();
      setGeneratedCode(result.response);
      setExplanation('Generated from simple query');
      setDependencies([]);
      setWarnings([]);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate response');
      console.error('Error generating response:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading && !generatedCode) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2">Loading...</span>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Performance Dashboard */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4">Performance Dashboard</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {performanceMetrics && (
            <>
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="font-semibold text-blue-800">Total Requests</h3>
                <p className="text-2xl font-bold text-blue-600">{performanceMetrics.total_requests}</p>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <h3 className="font-semibold text-green-800">Avg Latency</h3>
                <p className="text-2xl font-bold text-green-600">{(performanceMetrics.avg_latency * 1000).toFixed(1)}ms</p>
              </div>
              <div className="bg-yellow-50 p-4 rounded-lg">
                <h3 className="font-semibold text-yellow-800">Throughput</h3>
                <p className="text-2xl font-bold text-yellow-600">{performanceMetrics.avg_throughput.toFixed(1)} req/s</p>
              </div>
              <div className="bg-purple-50 p-4 rounded-lg">
                <h3 className="font-semibold text-purple-800">Total Tokens</h3>
                <p className="text-2xl font-bold text-purple-600">{performanceMetrics.total_tokens}</p>
              </div>
            </>
          )}
        </div>
        
        {modelInfo && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <h3 className="font-semibold mb-2">Model Information</h3>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div><span className="font-medium">Model:</span> {modelInfo.model_name}</div>
              <div><span className="font-medium">Type:</span> {modelInfo.model_type}</div>
              <div><span className="font-medium">Fine-tuned:</span> {modelInfo.fine_tuned ? 'Yes' : 'No'}</div>
              <div><span className="font-medium">vLLM:</span> {modelInfo.vllm_enabled ? 'Enabled' : 'Disabled'}</div>
            </div>
          </div>
        )}
      </div>

      {/* Simple Query Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4">Simple Query</h2>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Ask a question about microcontroller programming..."
            className="flex-1 p-2 border border-gray-300 rounded-md"
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                const target = e.target as HTMLInputElement;
                handleSimpleQuery(target.value);
              }
            }}
          />
          <button
            onClick={() => {
              const input = document.querySelector('input[placeholder*="Ask a question"]') as HTMLInputElement;
              if (input?.value) {
                handleSimpleQuery(input.value);
              }
            }}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Ask
          </button>
        </div>
      </div>

      {/* Dynamic Code Generation Form */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4">Dynamic Code Generation</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Peripheral
              </label>
              <select
                value={selectedPeripheral}
                onChange={(e) => handlePeripheralChange(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md"
              >
                <option value="">Select a peripheral</option>
                {Array.isArray(peripherals) && peripherals.length > 0 ? (
                  peripherals.map(peripheral => (
                    <option key={peripheral} value={peripheral}>
                      {peripheral.toUpperCase()}
                    </option>
                  ))
                ) : (
                  <>
                    <option value="uart">UART</option>
                    <option value="spi">SPI</option>
                    <option value="i2c">I2C</option>
                    <option value="gpio">GPIO</option>
                  </>
                )}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Operation
              </label>
              <select
                value={selectedOperation}
                onChange={(e) => handleOperationChange(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md"
                disabled={!selectedPeripheral}
              >
                <option value="">Select an operation</option>
                {operations.map(operation => (
                  <option key={operation} value={operation}>
                    {operation.replace(/_/g, ' ').toUpperCase()}
                  </option>
                ))}
              </select>
              {peripheralData && selectedOperation && (
                <p className="text-sm text-gray-600 mt-1">
                  {peripheralData.descriptions[selectedOperation]}
                </p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Language
              </label>
              <select
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md"
              >
                {languages.map(language => (
                  <option key={language} value={language}>
                    {language.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-2">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={includeComments}
                  onChange={(e) => setIncludeComments(e.target.checked)}
                  className="mr-2"
                />
                <span className="text-sm font-medium text-gray-700">Include Comments</span>
              </label>
              
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={includeErrorHandling}
                  onChange={(e) => setIncludeErrorHandling(e.target.checked)}
                  className="mr-2"
                />
                <span className="text-sm font-medium text-gray-700">Include Error Handling</span>
              </label>
            </div>
          </div>

          {/* Right Column - Parameters */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Parameters
            </label>
            <div className="space-y-3">
              {Object.keys(parameters).map(paramName => (
                <div key={paramName}>
                  <label className="block text-xs font-medium text-gray-600 mb-1">
                    {paramName.replace(/_/g, ' ').toUpperCase()}
                  </label>
                  <input
                    type="text"
                    value={parameters[paramName]}
                    onChange={(e) => handleParameterChange(paramName, e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded-md text-sm"
                    placeholder={`Enter ${paramName}`}
                  />
                </div>
              ))}
              {Object.keys(parameters).length === 0 && (
                <p className="text-sm text-gray-500 italic">
                  Select an operation to see required parameters
                </p>
              )}
            </div>
          </div>
        </div>

        <div className="mt-6">
          <button
            onClick={generateCode}
            disabled={!selectedPeripheral || !selectedOperation || loading}
            className="w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {loading ? 'Generating...' : 'Generate Code'}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* Generated Code Display */}
      {generatedCode && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-2xl font-bold mb-4">Generated Code</h2>
          
          {explanation && (
            <div className="mb-4 p-3 bg-blue-50 rounded-md">
              <p className="text-blue-800">{explanation}</p>
            </div>
          )}

          <div className="mb-4">
            <pre className="bg-gray-900 text-green-400 p-4 rounded-md overflow-x-auto">
              <code>{generatedCode}</code>
            </pre>
          </div>

          {dependencies.length > 0 && (
            <div className="mb-4">
              <h3 className="font-semibold text-gray-700 mb-2">Dependencies:</h3>
              <ul className="list-disc list-inside text-sm text-gray-600">
                {dependencies.map((dep, index) => (
                  <li key={index}>{dep}</li>
                ))}
              </ul>
            </div>
          )}

          {warnings.length > 0 && (
            <div>
              <h3 className="font-semibold text-yellow-700 mb-2">Warnings:</h3>
              <ul className="list-disc list-inside text-sm text-yellow-600">
                {warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DynamicCodeGenerator;
