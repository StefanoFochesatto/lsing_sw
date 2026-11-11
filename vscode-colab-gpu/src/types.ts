export interface ColabSession {
  name: string;
  token: string;
  url: string;
  endpoint: string;
  tokenExpiresAt?: string | null;
  variant: string; // "DEFAULT", "GPU", "TPU"
  accelerator: string; // "NONE", "T4", "L4", "A100", "H100", "v5e1", "v6e1"
  machineShape: string; // "STANDARD", "HIGH_MEM"
  kernelId?: string | null;
  sessionId?: string | null;
  lastExecution?: [string, string | null, string] | null; // [file, cell, timestamp]
  running?: string | null;
  status?: string; // "IDLE", "BUSY (file)", "UNKNOWN"
}

export interface ColabUsageInfo {
  balance: number;
  hourlyRate: number;
  activeAssignments: number;
  rawOutput?: string;
  authenticated: boolean;
  error?: string;
}

export interface ColabLogEntry {
  timestamp?: string;
  type?: string;
  message?: string;
  raw?: string;
}
