import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { ColabSession, ColabUsageInfo } from './types';

const execFileAsync = promisify(execFile);

interface RawSessionRecord {
  name?: string;
  token?: string;
  url?: string;
  endpoint?: string;
  token_expires_at?: string | null;
  variant?: string;
  accelerator?: string;
  machine_shape?: string;
  kernel_id?: string | null;
  session_id?: string | null;
  last_execution?: [string, string | null, string] | null;
  running?: string | null;
}

function getErrorMessage(error: unknown): string {
  if (!error || typeof error !== 'object') {
    return String(error);
  }
  if ('stderr' in error && typeof error.stderr === 'string') {
    return error.stderr;
  }
  if ('message' in error && typeof error.message === 'string') {
    return error.message;
  }
  return String(error);
}

export class ColabService {
  private _onDidChangeSessions = new vscode.EventEmitter<void>();
  readonly onDidChangeSessions: vscode.Event<void> = this._onDidChangeSessions.event;

  private _fileWatcher?: fs.FSWatcher;
  private _configDir = path.join(os.homedir(), '.config', 'colab-cli');
  private _sessionsFile = path.join(this._configDir, 'sessions.json');
  private _tokenFile = path.join(this._configDir, 'token.json');

  constructor() {
    this._setupFileWatcher();
  }

  public getCliPath(): string {
    const config = vscode.workspace.getConfiguration('colab');
    const customPath = config.get<string>('cliPath');
    if (customPath && fs.existsSync(customPath)) {
      return customPath;
    }

    const defaultUvPath = path.join(os.homedir(), '.local', 'bin', 'colab');
    if (fs.existsSync(defaultUvPath)) {
      return defaultUvPath;
    }

    return 'colab';
  }

  public isAuthenticated(): boolean {
    return fs.existsSync(this._tokenFile);
  }

  private _setupFileWatcher() {
    try {
      if (!fs.existsSync(this._configDir)) {
        fs.mkdirSync(this._configDir, { recursive: true });
      }

      this._fileWatcher = fs.watch(this._configDir, (_eventType: fs.WatchEventType, filename: string | null) => {
        if (filename === 'sessions.json' || filename === 'token.json') {
          this._onDidChangeSessions.fire();
        }
      });
    } catch (e) {
      console.warn('ColabService: Could not initialize file watcher on config dir', e);
    }
  }

  /**
   * Reads sessions quickly from local sessions.json
   */
  public async getLocalSessions(): Promise<ColabSession[]> {
    try {
      if (!fs.existsSync(this._sessionsFile)) {
        return [];
      }
      const raw = await fs.promises.readFile(this._sessionsFile, 'utf-8');
      if (!raw || !raw.trim()) {
        return [];
      }
      const data = JSON.parse(raw) as Record<string, RawSessionRecord>;
      const sessions: ColabSession[] = [];

      for (const [key, val] of Object.entries(data)) {
        const accelerator = val.accelerator === 'NONE' ? 'CPU' : val.accelerator || 'CPU';
        const status = val.running ? `BUSY (${val.running})` : 'IDLE';

        sessions.push({
          name: val.name || key,
          token: val.token || '',
          url: val.url || '',
          endpoint: val.endpoint || '',
          tokenExpiresAt: val.token_expires_at,
          variant: val.variant || 'DEFAULT',
          accelerator: accelerator,
          machineShape: val.machine_shape || 'STANDARD',
          kernelId: val.kernel_id,
          sessionId: val.session_id,
          lastExecution: val.last_execution,
          running: val.running,
          status: status,
        });
      }

      return sessions;
    } catch (err) {
      console.error('ColabService: Error reading local sessions', err);
      return [];
    }
  }

  /**
   * Syncs and retrieves server-side sessions via `colab sessions`
   */
  public async syncServerSessions(): Promise<{ sessions: ColabSession[]; rawOutput: string }> {
    const cli = this.getCliPath();
    try {
      const { stdout } = await execFileAsync(cli, ['sessions'], {
        timeout: 15000,
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      });

      const sessions = await this.getLocalSessions();
      return { sessions, rawOutput: stdout.trim() };
    } catch (error: unknown) {
      const stderr = getErrorMessage(error);
      if (stderr.includes('accounts.google.com') || stderr.includes('authorize colab-cli')) {
        throw new Error('AUTH_REQUIRED');
      }
      const sessions = await this.getLocalSessions();
      return { sessions, rawOutput: stderr };
    }
  }

  /**
   * Fetches compute-unit balance and consumption rate via `colab usage`
   */
  public async getUsage(): Promise<ColabUsageInfo> {
    const cli = this.getCliPath();
    try {
      const { stdout } = await execFileAsync(cli, ['usage'], {
        timeout: 15000,
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      });

      const text = stdout.trim();
      let balance = 0;
      let hourlyRate = 0;
      let activeAssignments = 0;

      const balanceMatch = text.match(/Current balance:\s*([\d.]+)/i);
      if (balanceMatch) {
        balance = parseFloat(balanceMatch[1]);
      }

      const rateMatch = text.match(/Usage rate:\s*([\d.]+)\/hr/i);
      if (rateMatch) {
        hourlyRate = parseFloat(rateMatch[1]);
      }

      const countMatch = text.match(/Active assignments:\s*([\d]+)/i);
      if (countMatch) {
        activeAssignments = parseInt(countMatch[1], 10);
      }

      return {
        balance,
        hourlyRate,
        activeAssignments,
        rawOutput: text,
        authenticated: true,
      };
    } catch (error: unknown) {
      const stderr = getErrorMessage(error);
      if (stderr.includes('accounts.google.com') || stderr.includes('authorize colab-cli')) {
        return {
          balance: 0,
          hourlyRate: 0,
          activeAssignments: 0,
          authenticated: false,
          error: 'Authentication required. Click to sign in with your Google Colab Pro account.',
        };
      }

      return {
        balance: 0,
        hourlyRate: 0,
        activeAssignments: 0,
        authenticated: this.isAuthenticated(),
        error: stderr,
      };
    }
  }

  /**
   * Provisions a new Colab session
   */
  public async createSession(
    name: string,
    accelerator?: 'T4' | 'L4' | 'A100' | 'CPU',
    highMem: boolean = false
  ): Promise<void> {
    const cli = this.getCliPath();
    const args = ['new', '-s', name];

    if (accelerator && accelerator !== 'CPU') {
      args.push('--gpu', accelerator);
    }
    if (highMem) {
      args.push('--high-mem');
    }

    await execFileAsync(cli, args, {
      timeout: 60000,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    this._onDidChangeSessions.fire();
  }

  /**
   * Stops an active session
   */
  public async stopSession(name: string): Promise<void> {
    const cli = this.getCliPath();
    await execFileAsync(cli, ['stop', '-s', name], {
      timeout: 30000,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });
    this._onDidChangeSessions.fire();
  }

  /**
   * Restarts the kernel of an active session
   */
  public async restartKernel(name: string): Promise<void> {
    const cli = this.getCliPath();
    await execFileAsync(cli, ['restart-kernel', '-s', name], {
      timeout: 30000,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });
    this._onDidChangeSessions.fire();
  }

  /**
   * Retrieves the browser URL to connect to the session
   */
  public async getSessionUrl(name: string): Promise<string> {
    const cli = this.getCliPath();
    const { stdout } = await execFileAsync(cli, ['url', '-s', name], {
      timeout: 10000,
    });
    const urlMatch = stdout.match(/https:\/\/[^\s]+/);
    return urlMatch ? urlMatch[0] : stdout.trim();
  }

  /**
   * Retrieves structured history logs for a session
   */
  public async getLogs(name: string, count: number = 30): Promise<string> {
    const cli = this.getCliPath();
    try {
      const { stdout } = await execFileAsync(cli, ['log', '-s', name, '-n', count.toString()], {
        timeout: 10000,
      });
      return stdout.trim();
    } catch (e: unknown) {
      return getErrorMessage(e) || 'No logs available.';
    }
  }

  public dispose() {
    if (this._fileWatcher) {
      this._fileWatcher.close();
    }
    this._onDidChangeSessions.dispose();
  }
}
