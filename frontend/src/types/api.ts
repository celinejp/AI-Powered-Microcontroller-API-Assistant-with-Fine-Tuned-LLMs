export interface CodeGenerationRequest {
  peripheral: string;
  microcontroller: string;
  operation: string;
  parameters?: Record<string, any>;
  sdk_version?: string;
  language: string;
  include_comments: boolean;
  include_error_handling: boolean;
}

export interface CodeGenerationResponse {
  code: string;
  explanation: string;
  peripheral: string;
  microcontroller: string;
  operation: string;
  language: string;
  sdk_compliance: string;
  dependencies: string[];
  warnings: string[];
}

export interface HealthCheckResponse {
  status: string;
  version: string;
  model_loaded: boolean;
  model_name?: string;
  uptime: number;
}

export interface ModelInfoResponse {
  model_name: string;
  model_type: string;
  parameters: number;
  max_length: number;
  supported_peripherals: string[];
  supported_microcontrollers: string[];
  fine_tuned: boolean;
  training_date?: string;
}

export interface SupportedOperationsResponse {
  peripheral: string;
  operations: string[];
  descriptions: Record<string, string>;
  required_parameters: Record<string, string[]>;
}

export interface ErrorResponse {
  error: string;
  detail?: string;
  error_code?: string;
}

export type PeripheralType = 'uart' | 'spi' | 'gpio' | 'i2c';
export type MicrocontrollerType = 'stm32' | 'esp32' | 'arduino' | 'raspberry_pi_pico' | 'nordic_nrf' | 'ti_msp430' | 'atmel_avr';
export type LanguageType = 'c' | 'cpp' | 'python';

// New simple query models for the updated /generate endpoint
export interface SimpleQueryRequest {
  query: string;
}

export interface SimpleQueryResponse {
  response: string;
  query: string;
  model_used: string;
}
