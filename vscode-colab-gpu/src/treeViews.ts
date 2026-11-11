import * as vscode from 'vscode';
import { ColabService } from './colabService';
import { ColabSession, ColabUsageInfo } from './types';

export class SessionTreeItem extends vscode.TreeItem {
  constructor(
    public readonly session: ColabSession,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState
  ) {
    super(session.name, collapsibleState);

    const isBusy = session.status?.startsWith('BUSY');
    const isGpu = session.accelerator !== 'CPU' && session.accelerator !== 'NONE';

    this.description = `${session.accelerator} [${session.machineShape}] - ${session.status}`;
    this.tooltip = `Colab Session: ${session.name}\nHardware: ${session.accelerator}\nShape: ${session.machineShape}\nStatus: ${session.status}\nEndpoint: ${session.endpoint}`;
    this.contextValue = 'session';

    if (isBusy) {
      this.iconPath = new vscode.ThemeIcon('sync~spin', new vscode.ThemeColor('charts.orange'));
    } else if (isGpu) {
      this.iconPath = new vscode.ThemeIcon('chip', new vscode.ThemeColor('charts.green'));
    } else {
      this.iconPath = new vscode.ThemeIcon('server-process');
    }
  }
}

export class DetailTreeItem extends vscode.TreeItem {
  constructor(label: string, value: string, icon: string, copyableValue?: string) {
    super(label, vscode.TreeItemCollapsibleState.None);
    this.description = value;
    this.tooltip = copyableValue ? `${label}: ${copyableValue} (Click to copy)` : `${label}: ${value}`;
    this.iconPath = new vscode.ThemeIcon(icon);

    if (copyableValue) {
      this.command = {
        command: 'vscode.env.clipboard.writeText',
        title: 'Copy to clipboard',
        arguments: [copyableValue],
      };
    }
  }
}

export class SessionsTreeDataProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<vscode.TreeItem | undefined | void>();
  readonly onDidChangeTreeData: vscode.Event<vscode.TreeItem | undefined | void> = this._onDidChangeTreeData.event;

  private _sessions: ColabSession[] = [];
  private _loading: boolean = false;
  private _authRequired: boolean = false;

  constructor(private readonly colabService: ColabService) {
    this.colabService.onDidChangeSessions(() => this.refresh());
  }

  public refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: vscode.TreeItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: vscode.TreeItem): Promise<vscode.TreeItem[]> {
    if (!element) {
      // Root level: fetch sessions
      try {
        this._loading = true;
        this._sessions = await this.colabService.getLocalSessions();
        this._authRequired = false;

        if (this._sessions.length === 0) {
          if (!this.colabService.isAuthenticated()) {
            const authItem = new vscode.TreeItem(
              'Sign in with Google Colab Pro',
              vscode.TreeItemCollapsibleState.None
            );
            authItem.description = 'Click to authenticate';
            authItem.iconPath = new vscode.ThemeIcon('key', new vscode.ThemeColor('charts.blue'));
            authItem.command = {
              command: 'colab.login',
              title: 'Authenticate Colab CLI',
            };
            return [authItem];
          }

          const emptyItem = new vscode.TreeItem(
            'No active Colab sessions',
            vscode.TreeItemCollapsibleState.None
          );
          emptyItem.description = 'Click "+" to launch a GPU VM';
          emptyItem.iconPath = new vscode.ThemeIcon('info');
          return [emptyItem];
        }

        return this._sessions.map(
          (s) => new SessionTreeItem(s, vscode.TreeItemCollapsibleState.Collapsed)
        );
      } catch (err: unknown) {
        const errorItem = new vscode.TreeItem(
          'Error loading sessions',
          vscode.TreeItemCollapsibleState.None
        );
        errorItem.description = String(err);
        errorItem.iconPath = new vscode.ThemeIcon('error');
        return [errorItem];
      } finally {
        this._loading = false;
      }
    }

    // Detail level for a session
    if (element instanceof SessionTreeItem) {
      const s = element.session;
      const details: DetailTreeItem[] = [];

      details.push(new DetailTreeItem('Status', s.status || 'IDLE', s.status?.startsWith('BUSY') ? 'sync~spin' : 'pass'));
      details.push(new DetailTreeItem('Hardware', s.accelerator, 'chip'));
      details.push(new DetailTreeItem('Machine Shape', s.machineShape, 'server'));

      if (s.endpoint) {
        details.push(new DetailTreeItem('Endpoint', s.endpoint, 'link', s.endpoint));
      }
      if (s.lastExecution) {
        const [file, cell, time] = s.lastExecution;
        const cellInfo = cell ? ` (Cell: ${cell})` : '';
        details.push(new DetailTreeItem('Last Run', `${file}${cellInfo} at ${time}`, 'play'));
      }
      if (s.kernelId) {
        details.push(new DetailTreeItem('Kernel ID', s.kernelId.slice(0, 12) + '...', 'symbol-numeric', s.kernelId));
      }
      if (s.tokenExpiresAt) {
        details.push(new DetailTreeItem('Token Expires', s.tokenExpiresAt, 'clock'));
      }

      return details;
    }

    return [];
  }
}

export class UsageTreeDataProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<vscode.TreeItem | undefined | void>();
  readonly onDidChangeTreeData: vscode.Event<vscode.TreeItem | undefined | void> = this._onDidChangeTreeData.event;

  private _usage?: ColabUsageInfo;

  constructor(private readonly colabService: ColabService) {
    this.colabService.onDidChangeSessions(() => this.refresh());
  }

  public refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: vscode.TreeItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: vscode.TreeItem): Promise<vscode.TreeItem[]> {
    if (element) {
      return [];
    }

    try {
      this._usage = await this.colabService.getUsage();

      if (!this._usage.authenticated) {
        const authItem = new vscode.TreeItem(
          'Authentication Required',
          vscode.TreeItemCollapsibleState.None
        );
        authItem.description = 'Sign in to view compute units';
        authItem.iconPath = new vscode.ThemeIcon('key', new vscode.ThemeColor('charts.yellow'));
        authItem.command = {
          command: 'colab.login',
          title: 'Authenticate Colab CLI',
        };
        return [authItem];
      }

      const items: vscode.TreeItem[] = [];

      // Balance
      const balanceItem = new vscode.TreeItem(
        `Balance: ${this._usage.balance.toFixed(2)} CCU`,
        vscode.TreeItemCollapsibleState.None
      );
      balanceItem.description = 'Compute Units remaining';
      balanceItem.iconPath = new vscode.ThemeIcon('credit-card', new vscode.ThemeColor('charts.green'));
      items.push(balanceItem);

      // Hourly Rate
      const rateItem = new vscode.TreeItem(
        `Usage Rate: ${this._usage.hourlyRate.toFixed(2)} CCU / hr`,
        vscode.TreeItemCollapsibleState.None
      );
      rateItem.description = this._usage.hourlyRate > 0 ? 'Active consumption' : 'No burning units';
      rateItem.iconPath = new vscode.ThemeIcon(
        'flame',
        this._usage.hourlyRate > 0 ? new vscode.ThemeColor('charts.orange') : undefined
      );
      items.push(rateItem);

      // Active Assignments
      const countItem = new vscode.TreeItem(
        `Active VMs: ${this._usage.activeAssignments}`,
        vscode.TreeItemCollapsibleState.None
      );
      countItem.description = this._usage.activeAssignments === 1 ? '1 VM running' : `${this._usage.activeAssignments} VMs running`;
      countItem.iconPath = new vscode.ThemeIcon('vm');
      items.push(countItem);

      // Remaining estimate
      if (this._usage.hourlyRate > 0) {
        const remainingHours = (this._usage.balance / this._usage.hourlyRate).toFixed(1);
        const estItem = new vscode.TreeItem(
          `Estimated Runtime: ~${remainingHours} hrs`,
          vscode.TreeItemCollapsibleState.None
        );
        estItem.description = 'At current consumption rate';
        estItem.iconPath = new vscode.ThemeIcon('clock');
        items.push(estItem);
      }

      return items;
    } catch (e: unknown) {
      const errItem = new vscode.TreeItem('Unable to fetch compute units', vscode.TreeItemCollapsibleState.None);
      errItem.description = String(e);
      errItem.iconPath = new vscode.ThemeIcon('warning');
      return [errItem];
    }
  }
}

export class ActionsTreeDataProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
  getTreeItem(element: vscode.TreeItem): vscode.TreeItem {
    return element;
  }

  async getChildren(): Promise<vscode.TreeItem[]> {
    const actions: vscode.TreeItem[] = [];

    const t4 = new vscode.TreeItem('Launch T4 GPU Session', vscode.TreeItemCollapsibleState.None);
    t4.description = '~1.5-2 CCU/hr';
    t4.iconPath = new vscode.ThemeIcon('plus', new vscode.ThemeColor('charts.blue'));
    t4.command = { command: 'colab.newSessionT4', title: 'Launch T4' };
    actions.push(t4);

    const l4 = new vscode.TreeItem('Launch L4 GPU Session', vscode.TreeItemCollapsibleState.None);
    l4.description = '~2-3 CCU/hr (Ada Lovelace)';
    l4.iconPath = new vscode.ThemeIcon('plus', new vscode.ThemeColor('charts.purple'));
    l4.command = { command: 'colab.newSessionL4', title: 'Launch L4' };
    actions.push(l4);

    const a100 = new vscode.TreeItem('Launch A100 GPU Session', vscode.TreeItemCollapsibleState.None);
    a100.description = '~10-13 CCU/hr (High-RAM)';
    a100.iconPath = new vscode.ThemeIcon('rocket', new vscode.ThemeColor('charts.red'));
    a100.command = { command: 'colab.newSessionA100', title: 'Launch A100' };
    actions.push(a100);

    const custom = new vscode.TreeItem('New Custom Session...', vscode.TreeItemCollapsibleState.None);
    custom.description = 'Choose name, GPU, TPU, High-RAM';
    custom.iconPath = new vscode.ThemeIcon('add');
    custom.command = { command: 'colab.newSessionCustom', title: 'New Custom Session' };
    actions.push(custom);

    const dashboard = new vscode.TreeItem('Open Visual Dashboard', vscode.TreeItemCollapsibleState.None);
    dashboard.description = 'Colab control center';
    dashboard.iconPath = new vscode.ThemeIcon('dashboard');
    dashboard.command = { command: 'colab.openDashboard', title: 'Open Dashboard' };
    actions.push(dashboard);

    const web = new vscode.TreeItem('Open Colab in Browser', vscode.TreeItemCollapsibleState.None);
    web.description = 'colab.research.google.com';
    web.iconPath = new vscode.ThemeIcon('globe');
    web.command = { command: 'colab.openWeb', title: 'Open Web' };
    actions.push(web);

    const login = new vscode.TreeItem('Authenticate Colab CLI...', vscode.TreeItemCollapsibleState.None);
    login.description = 'Sign in with Google';
    login.iconPath = new vscode.ThemeIcon('key');
    login.command = { command: 'colab.login', title: 'Authenticate' };
    actions.push(login);

    return actions;
  }
}
