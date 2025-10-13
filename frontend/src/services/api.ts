import axios, { AxiosResponse } from 'axios';
import {
  CodeGenerationRequest,
  CodeGenerationResponse,
  HealthCheckResponse,
  ModelInfoResponse,
  SupportedOperationsResponse,
  SimpleQueryRequest,
  SimpleQueryResponse,
  // ErrorResponse, // Unused - commented out
  // PeripheralType, // Unused - commented out
  // MicrocontrollerType, // Unused - commented out
  // LanguageType, // Unused - commented out
} from '../types/api';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response: AxiosResponse) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error);
    if (error.response?.data) {
      console.error('Error details:', error.response.data);
    }
    return Promise.reject(error);
  }
);

export const apiService = {
  // Health check
  async getHealth(): Promise<HealthCheckResponse> {
    const response = await api.get<HealthCheckResponse>('/health');
    return response.data;
  },

  // Model information
  async getModelInfo(): Promise<ModelInfoResponse> {
    const response = await api.get<ModelInfoResponse>('/model/info');
    return response.data;
  },

  // Get supported peripherals
  async getSupportedPeripherals(): Promise<string[]> {
    const response = await api.get<string[]>('/peripherals');
    return response.data;
  },

  // Get supported microcontrollers
  async getSupportedMicrocontrollers(): Promise<string[]> {
    const response = await api.get<string[]>('/microcontrollers');
    return response.data;
  },

  // Get supported operations for a peripheral
  async getSupportedOperations(peripheral: string): Promise<SupportedOperationsResponse> {
    const response = await api.get<SupportedOperationsResponse>(`/operations/${peripheral}`);
    return response.data;
  },

  // Get supported languages
  async getSupportedLanguages(): Promise<string[]> {
    const response = await api.get<string[]>('/languages');
    return response.data;
  },

  // Generate code (legacy endpoint)
  async generateCode(request: CodeGenerationRequest): Promise<CodeGenerationResponse> {
    const response = await api.post<CodeGenerationResponse>('/generate-code', request);
    return response.data;
  },

  // Generate response for simple query (new endpoint)
  async generateSimpleQuery(request: SimpleQueryRequest): Promise<SimpleQueryResponse> {
    const response = await api.post<SimpleQueryResponse>('/generate', request);
    return response.data;
  },
};

export default apiService;
