import * as vscode from 'vscode';
import { ColabService } from './colabService';
import { ColabSession, ColabUsageInfo } from './types';

interface DashboardMessage {
  command: string;
  sessionName?: string;
  accelerator?: 'T4' | 'L4' | 'A100' | 'CPU';
  highMem?: boolean;
}

export class ColabDashboardPanel {
  public static currentPanel: ColabDashboardPanel | undefined;
  private readonly _panel: vscode.WebviewPanel;
  private readonly _colabService: ColabService;
  private _disposables: vscode.Disposable[] = [];

  public static createOrShow(colabService: ColabService, extensionUri: vscode.Uri) {
    const column = vscode.window.activeTextEditor
      ? vscode.window.activeTextEditor.viewColumn
      : undefined;

    if (ColabDashboardPanel.currentPanel) {
      ColabDashboardPanel.currentPanel._panel.reveal(column);
      ColabDashboardPanel.currentPanel.update();
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      'colabDashboard',
      'Google Colab GPU Dashboard',
      column || vscode.ViewColumn.One,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [extensionUri],
      }
    );

    ColabDashboardPanel.currentPanel = new ColabDashboardPanel(panel, colabService);
  }

  private constructor(panel: vscode.WebviewPanel, colabService: ColabService) {
    this._panel = panel;
    this._colabService = colabService;

    this.update();

    this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

    this._panel.webview.onDidReceiveMessage(
      async (message: DashboardMessage) => {
        switch (message.command) {
          case 'refresh':
            await this.update();
            vscode.window.showInformationMessage('Colab sessions refreshed.');
            return;
          case 'ssh':
            if (message.sessionName) {
              vscode.commands.executeCommand('colab.sshConnect', { session: { name: message.sessionName } });
            }
            return;
          case 'web':
            if (message.sessionName) {
              vscode.commands.executeCommand('colab.openWeb', { session: { name: message.sessionName } });
            }
            return;
          case 'stop':
            if (message.sessionName) {
              vscode.commands.executeCommand('colab.stopSession', { session: { name: message.sessionName } });
            }
            return;
          case 'restart':
            if (message.sessionName) {
              vscode.commands.executeCommand('colab.restartKernel', { session: { name: message.sessionName } });
            }
            return;
          case 'logs':
            if (message.sessionName) {
              vscode.commands.executeCommand('colab.viewLogs', { session: { name: message.sessionName } });
            }
            return;
          case 'launch':
            if (message.accelerator) {
              const sessionName = `gpu-${message.accelerator.toLowerCase()}-${Date.now().toString().slice(-4)}`;
              await vscode.window.withProgress(
                {
                  location: vscode.ProgressLocation.Notification,
                  title: `Provisioning ${message.accelerator} GPU runtime on Google Colab...`,
                  cancellable: false,
                },
                async () => {
                  await this._colabService.createSession(sessionName, message.accelerator, message.highMem || false);
                }
              );
              await this.update();
            }
            return;
          case 'login':
            vscode.commands.executeCommand('colab.login');
            return;
        }
      },
      null,
      this._disposables
    );

    this._colabService.onDidChangeSessions(() => {
      this.update();
    }, null, this._disposables);
  }

  public async update() {
    const sessions = await this._colabService.getLocalSessions();
    const usage = await this._colabService.getUsage();
    this._panel.webview.html = this._getHtmlForWebview(sessions, usage);
  }

  private _getHtmlForWebview(sessions: ColabSession[], usage: ColabUsageInfo): string {
    const sessionCards = sessions.map((s) => {
      const isBusy = s.status?.startsWith('BUSY');
      const statusBadge = isBusy
        ? `<span class="badge badge-busy">BUSY</span>`
        : `<span class="badge badge-idle">IDLE</span>`;

      const accelBadgeClass =
        s.accelerator === 'A100' ? 'badge-a100' :
        s.accelerator === 'L4' ? 'badge-l4' :
        s.accelerator === 'T4' ? 'badge-t4' : 'badge-cpu';

      const lastRun = s.lastExecution
        ? `<div class="detail-row"><span class="label">Last Exec:</span> <span class="val">${s.lastExecution[0]} at ${s.lastExecution[2]}</span></div>`
        : '';

      return `
        <div class="session-card">
          <div class="card-header">
            <div class="card-title">
              <span class="session-icon">⚡</span>
              <h3>${s.name}</h3>
            </div>
            <div class="badges">
              <span class="badge ${accelBadgeClass}">${s.accelerator}</span>
              <span class="badge badge-shape">${s.machineShape}</span>
              ${statusBadge}
            </div>
          </div>
          <div class="card-body">
            <div class="detail-row"><span class="label">Endpoint:</span> <span class="val code">${s.endpoint || 'N/A'}</span></div>
            ${s.kernelId ? `<div class="detail-row"><span class="label">Kernel ID:</span> <span class="val code">${s.kernelId.slice(0, 16)}...</span></div>` : ''}
            ${lastRun}
          </div>
          <div class="card-actions">
            <button class="btn btn-primary" onclick="vscode.postMessage({command: 'ssh', sessionName: '${s.name}'})">
              SSH Terminal
            </button>
            <button class="btn btn-secondary" onclick="vscode.postMessage({command: 'web', sessionName: '${s.name}'})">
              Colab Web
            </button>
            <button class="btn btn-secondary" onclick="vscode.postMessage({command: 'logs', sessionName: '${s.name}'})">
              Logs
            </button>
            <button class="btn btn-secondary" onclick="vscode.postMessage({command: 'restart', sessionName: '${s.name}'})">
              Restart
            </button>
            <button class="btn btn-danger" onclick="vscode.postMessage({command: 'stop', sessionName: '${s.name}'})">
              Stop VM
            </button>
          </div>
        </div>
      `;
    }).join('');

    const emptySessionsHtml = `
      <div class="empty-state">
        <div class="empty-icon">☁️</div>
        <h3>No active Colab runtimes</h3>
        <p>Provision a new GPU instance below to begin running benchmarks.</p>
        <div class="quick-launch-buttons">
          <button class="btn btn-primary" onclick="vscode.postMessage({command: 'launch', accelerator: 'T4'})">
            + Launch T4 GPU (~1.5 CCU/hr)
          </button>
          <button class="btn btn-primary" onclick="vscode.postMessage({command: 'launch', accelerator: 'L4'})">
            + Launch L4 GPU (~2.5 CCU/hr)
          </button>
          <button class="btn btn-primary" onclick="vscode.postMessage({command: 'launch', accelerator: 'A100', highMem: true})">
            + Launch A100 High-RAM (~11 CCU/hr)
          </button>
        </div>
      </div>
    `;

    const remainingHours = usage.hourlyRate > 0
      ? `~${(usage.balance / usage.hourlyRate).toFixed(1)} hrs`
      : '∞ (No active burn)';

    return `<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Google Colab GPU Dashboard</title>
      <style>
        :root {
          --bg-color: var(--vscode-editor-background);
          --card-bg: var(--vscode-sideBar-background, #1e1e1e);
          --text-color: var(--vscode-editor-foreground);
          --border-color: var(--vscode-widget-border, rgba(255, 255, 255, 0.1));
          --accent-color: var(--vscode-button-background, #0e639c);
          --accent-hover: var(--vscode-button-hoverBackground, #1177bb);
        }
        body {
          font-family: var(--vscode-font-family, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif);
          background-color: var(--bg-color);
          color: var(--text-color);
          margin: 0;
          padding: 24px;
        }
        .header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 24px;
          border-bottom: 1px solid var(--border-color);
          padding-bottom: 16px;
        }
        .header-title {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .header-title h1 {
          margin: 0;
          font-size: 22px;
          font-weight: 600;
        }
        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 16px;
          margin-bottom: 28px;
        }
        .metric-card {
          background-color: var(--card-bg);
          border: 1px solid var(--border-color);
          border-radius: 8px;
          padding: 16px;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .metric-label {
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          opacity: 0.7;
        }
        .metric-value {
          font-size: 26px;
          font-weight: 700;
        }
        .metric-sub {
          font-size: 12px;
          opacity: 0.8;
        }
        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
        }
        .section-header h2 {
          font-size: 16px;
          margin: 0;
        }
        .sessions-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
          gap: 16px;
          margin-bottom: 28px;
        }
        .session-card {
          background-color: var(--card-bg);
          border: 1px solid var(--border-color);
          border-radius: 8px;
          padding: 18px;
          display: flex;
          flex-direction: column;
          gap: 14px;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .card-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
        }
        .card-title {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .card-title h3 {
          margin: 0;
          font-size: 16px;
        }
        .badges {
          display: flex;
          gap: 6px;
          flex-wrap: wrap;
        }
        .badge {
          font-size: 11px;
          font-weight: 600;
          padding: 2px 8px;
          border-radius: 4px;
        }
        .badge-a100 { background-color: #d32f2f; color: white; }
        .badge-l4 { background-color: #7b1fa2; color: white; }
        .badge-t4 { background-color: #1976d2; color: white; }
        .badge-cpu { background-color: #555; color: white; }
        .badge-shape { background-color: #333; color: #ddd; }
        .badge-idle { background-color: #388e3c; color: white; }
        .badge-busy { background-color: #f57c00; color: white; }
        .card-body {
          display: flex;
          flex-direction: column;
          gap: 6px;
          font-size: 13px;
        }
        .detail-row {
          display: flex;
          gap: 8px;
        }
        .detail-row .label { opacity: 0.7; min-width: 75px; }
        .detail-row .val { font-weight: 500; }
        .detail-row .val.code { font-family: monospace; font-size: 11px; }
        .card-actions {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
          margin-top: auto;
          padding-top: 8px;
          border-top: 1px solid var(--border-color);
        }
        .btn {
          border: none;
          border-radius: 4px;
          padding: 6px 12px;
          font-size: 12px;
          font-weight: 500;
          cursor: pointer;
          transition: background-color 0.15s ease;
        }
        .btn-primary {
          background-color: var(--accent-color);
          color: var(--vscode-button-foreground, white);
        }
        .btn-primary:hover { background-color: var(--accent-hover); }
        .btn-secondary {
          background-color: var(--vscode-button-secondaryBackground, #3a3d41);
          color: var(--vscode-button-secondaryForeground, white);
        }
        .btn-secondary:hover { background-color: var(--vscode-button-secondaryHoverBackground, #45494e); }
        .btn-danger {
          background-color: #c62828;
          color: white;
        }
        .btn-danger:hover { background-color: #b71c1c; }
        .empty-state {
          text-align: center;
          padding: 40px 20px;
          background-color: var(--card-bg);
          border: 1px dashed var(--border-color);
          border-radius: 8px;
          margin-bottom: 24px;
        }
        .empty-icon { font-size: 40px; margin-bottom: 12px; }
        .quick-launch-buttons {
          display: flex;
          justify-content: center;
          gap: 12px;
          margin-top: 20px;
          flex-wrap: wrap;
        }
        .auth-banner {
          background-color: rgba(234, 134, 0, 0.15);
          border: 1px solid #ea8600;
          border-radius: 6px;
          padding: 14px 18px;
          margin-bottom: 20px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
      </style>
    </head>
    <body>
      <div class="header">
        <div class="header-title">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
            <path d="M7.5 7C4.46 7 2 9.46 2 12.5S4.46 18 7.5 18c2.4 0 4.45-1.54 5.2-3.7-.58-.7-1.35-1.2-2.2-1.4C9.8 14.5 8.7 15.5 7.5 15.5 5.84 15.5 4.5 14.16 4.5 12.5S5.84 9.5 7.5 9.5c1.2 0 2.3 1 3 2.6.85-.2 1.62-.7 2.2-1.4C11.95 8.54 9.9 7 7.5 7z" fill="#f9ab00"/>
            <path d="M16.5 7c-2.4 0-4.45 1.54-5.2 3.7.58.7 1.35 1.2 2.2 1.4.7-1.6 1.8-2.6 3-2.6 1.66 0 3 1.34 3 3s-1.34 3-3 3c-1.2 0-2.3-1-3-2.6-.85.2-1.62.7-2.2 1.4.75 2.16 2.8 3.7 5.2 3.7 3.04 0 5.5-2.46 5.5-5.5S19.54 7 16.5 7z" fill="#e37400"/>
          </svg>
          <h1>Google Colab Pro GPU Monitor</h1>
        </div>
        <div>
          <button class="btn btn-secondary" onclick="vscode.postMessage({command: 'refresh'})">↻ Refresh</button>
        </div>
      </div>

      ${!usage.authenticated ? `
        <div class="auth-banner">
          <div>
            <strong>Google Account Sign-In Required</strong>
            <div style="font-size: 12px; margin-top: 4px;">Sign in to authenticate Colab CLI with your Colab Pro subscription.</div>
          </div>
          <button class="btn btn-primary" onclick="vscode.postMessage({command: 'login'})">Sign In</button>
        </div>
      ` : ''}

      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-label">Compute Units Balance</div>
          <div class="metric-value">${usage.balance.toFixed(2)} <span style="font-size: 14px; font-weight: normal;">CCU</span></div>
          <div class="metric-sub">${usage.authenticated ? 'Colab Pro Active' : 'Not Authenticated'}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Usage Rate</div>
          <div class="metric-value" style="color: ${usage.hourlyRate > 0 ? '#ff9800' : 'inherit'};">
            ${usage.hourlyRate.toFixed(2)} <span style="font-size: 14px; font-weight: normal;">/ hr</span>
          </div>
          <div class="metric-sub">${usage.hourlyRate > 0 ? 'Burning compute units' : 'Zero burn rate'}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Active GPU / TPU Instances</div>
          <div class="metric-value">${usage.activeAssignments}</div>
          <div class="metric-sub">${sessions.length} tracked locally</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Remaining Runtime Estimate</div>
          <div class="metric-value" style="font-size: 20px;">${remainingHours}</div>
          <div class="metric-sub">At current hourly rate</div>
        </div>
      </div>

      <div class="section-header">
        <h2>Active Sessions</h2>
      </div>

      ${sessions.length > 0 ? `<div class="sessions-grid">${sessionCards}</div>` : emptySessionsHtml}

      <div class="section-header" style="margin-top: 28px;">
        <h2>Quick Launch Hardware</h2>
      </div>
      <div class="quick-launch-buttons" style="justify-content: flex-start; margin-top: 0;">
        <button class="btn btn-primary" onclick="vscode.postMessage({command: 'launch', accelerator: 'T4'})">
          + Launch T4 GPU
        </button>
        <button class="btn btn-primary" onclick="vscode.postMessage({command: 'launch', accelerator: 'L4'})">
          + Launch L4 GPU
        </button>
        <button class="btn btn-primary" onclick="vscode.postMessage({command: 'launch', accelerator: 'A100', highMem: true})">
          + Launch A100 High-RAM
        </button>
        <button class="btn btn-secondary" onclick="vscode.postMessage({command: 'launch', accelerator: 'CPU'})">
          + Launch CPU VM
        </button>
      </div>

      <script>
        const vscode = acquireVsCodeApi();
      </script>
    </body>
    </html>`;
  }

  public dispose() {
    ColabDashboardPanel.currentPanel = undefined;
    this._panel.dispose();
    while (this._disposables.length) {
      const x = this._disposables.pop();
      if (x) {
        x.dispose();
      }
    }
  }
}
