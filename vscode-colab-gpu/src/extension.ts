import * as vscode from 'vscode';
import { ColabService } from './colabService';
import { SessionsTreeDataProvider, UsageTreeDataProvider, ActionsTreeDataProvider, SessionTreeItem } from './treeViews';
import { ColabDashboardPanel } from './dashboardPanel';
import { ColabSession } from './types';

let statusBarItem: vscode.StatusBarItem;
let autoRefreshTimer: NodeJS.Timeout | undefined;

function resolveSessionName(target?: unknown): string | undefined {
  if (target instanceof SessionTreeItem) {
    return target.session.name;
  }
  if (target && typeof target === 'object') {
    if ('session' in target && target.session && typeof target.session === 'object' && 'name' in target.session && typeof target.session.name === 'string') {
      return target.session.name;
    }
    if ('name' in target && typeof target.name === 'string') {
      return target.name;
    }
  }
  return undefined;
}

export function activate(context: vscode.ExtensionContext) {
  const colabService = new ColabService();
  context.subscriptions.push(colabService);

  const outputChannel = vscode.window.createOutputChannel('Google Colab');
  context.subscriptions.push(outputChannel);

  // Register Tree Views
  const sessionsProvider = new SessionsTreeDataProvider(colabService);
  const usageProvider = new UsageTreeDataProvider(colabService);
  const actionsProvider = new ActionsTreeDataProvider();

  context.subscriptions.push(
    vscode.window.registerTreeDataProvider('colab.sessionsView', sessionsProvider),
    vscode.window.registerTreeDataProvider('colab.usageView', usageProvider),
    vscode.window.registerTreeDataProvider('colab.actionsView', actionsProvider)
  );

  // Status Bar Item
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 90);
  statusBarItem.command = 'colab.openDashboard';
  context.subscriptions.push(statusBarItem);

  const updateStatusBar = async () => {
    const config = vscode.workspace.getConfiguration('colab');
    if (!config.get<boolean>('showStatusBarItem', true)) {
      statusBarItem.hide();
      return;
    }

    try {
      const sessions = await colabService.getLocalSessions();
      const usage = await colabService.getUsage();

      if (!usage.authenticated) {
        statusBarItem.text = '$(key) Colab: Sign In';
        statusBarItem.tooltip = 'Click to authenticate Google Colab Pro CLI';
        statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        statusBarItem.show();
        return;
      }

      const activeCount = sessions.length;
      if (activeCount > 0) {
        const primaryGpu = sessions[0].accelerator;
        statusBarItem.text = `$(flame) Colab: ${activeCount} active (${primaryGpu}) | ${usage.balance.toFixed(1)} CCU`;
        statusBarItem.tooltip = `Colab Pro: ${activeCount} active VM(s)\nBurn Rate: ${usage.hourlyRate.toFixed(2)} CCU/hr\nBalance: ${usage.balance.toFixed(2)} CCU\nClick to open Colab Dashboard`;
        statusBarItem.backgroundColor = usage.hourlyRate > 0 ? new vscode.ThemeColor('statusBarItem.warningBackground') : undefined;
      } else {
        statusBarItem.text = `$(server-process) Colab: Idle | ${usage.balance.toFixed(1)} CCU`;
        statusBarItem.tooltip = `Colab Pro: No active instances\nBalance: ${usage.balance.toFixed(2)} CCU\nClick to open Colab Dashboard`;
        statusBarItem.backgroundColor = undefined;
      }

      statusBarItem.show();
    } catch {
      statusBarItem.text = '$(server-process) Colab';
      statusBarItem.show();
    }
  };

  const refreshAll = async (notify: boolean = false) => {
    try {
      await colabService.syncServerSessions();
    } catch {
      // Ignore background sync errors
    }
    sessionsProvider.refresh();
    usageProvider.refresh();
    await updateStatusBar();
    if (notify) {
      vscode.window.showInformationMessage('Google Colab status and sessions updated.');
    }
  };

  // Setup periodic refresh
  const setupAutoRefresh = () => {
    if (autoRefreshTimer) {
      clearInterval(autoRefreshTimer);
      autoRefreshTimer = undefined;
    }
    const config = vscode.workspace.getConfiguration('colab');
    const intervalSec = config.get<number>('autoRefreshInterval', 20);
    if (intervalSec > 0) {
      autoRefreshTimer = setInterval(() => {
        refreshAll(false);
      }, intervalSec * 1000);
    }
  };

  setupAutoRefresh();
  vscode.workspace.onDidChangeConfiguration((e) => {
    if (e.affectsConfiguration('colab')) {
      setupAutoRefresh();
      updateStatusBar();
    }
  });

  // Commands
  context.subscriptions.push(
    vscode.commands.registerCommand('colab.refresh', async () => {
      await refreshAll(true);
    }),

    vscode.commands.registerCommand('colab.openDashboard', () => {
      ColabDashboardPanel.createOrShow(colabService, context.extensionUri);
    }),

    vscode.commands.registerCommand('colab.login', () => {
      const cli = colabService.getCliPath();
      const terminal = vscode.window.createTerminal({
        name: 'Colab Auth',
        message: 'Authenticating with Google Colab Pro...',
      });
      terminal.show();
      terminal.sendText(`${cli} sessions`);
    }),

    vscode.commands.registerCommand('colab.sshConnect', async (target?: unknown) => {
      let sessionName = resolveSessionName(target);

      if (!sessionName) {
        const sessions = await colabService.getLocalSessions();
        if (sessions.length === 0) {
          const launch = await vscode.window.showQuickPick(
            ['Launch T4 GPU and Connect', 'Launch L4 GPU and Connect', 'Launch A100 High-RAM and Connect', 'Cancel'],
            { placeHolder: 'No active Colab sessions. Provision a GPU runtime now?' }
          );
          if (launch === 'Launch T4 GPU and Connect') {
            sessionName = `colab-t4-${Date.now().toString().slice(-4)}`;
            await colabService.createSession(sessionName, 'T4');
          } else if (launch === 'Launch L4 GPU and Connect') {
            sessionName = `colab-l4-${Date.now().toString().slice(-4)}`;
            await colabService.createSession(sessionName, 'L4');
          } else if (launch === 'Launch A100 High-RAM and Connect') {
            sessionName = `colab-a100-${Date.now().toString().slice(-4)}`;
            await colabService.createSession(sessionName, 'A100', true);
          } else {
            return;
          }
        } else if (sessions.length === 1) {
          sessionName = sessions[0].name;
        } else {
          const picked = await vscode.window.showQuickPick(
            sessions.map((s) => ({
              label: s.name,
              description: `${s.accelerator} [${s.machineShape}] - ${s.status}`,
              session: s,
            })),
            { placeHolder: 'Select a Colab session to connect via SSH' }
          );
          if (!picked) {
            return;
          }
          sessionName = picked.session.name;
        }
      }

      const cli = colabService.getCliPath();
      const terminal = vscode.window.createTerminal({
        name: `Colab SSH: ${sessionName}`,
      });
      terminal.show();
      terminal.sendText(`${cli} ssh -s ${sessionName}`);
    }),

    vscode.commands.registerCommand('colab.openWeb', async (target?: unknown) => {
      let sessionName = resolveSessionName(target);
      if (!sessionName) {
        const sessions = await colabService.getLocalSessions();
        if (sessions.length === 0) {
          vscode.env.openExternal(vscode.Uri.parse('https://colab.research.google.com'));
          return;
        }
        sessionName = sessions[0].name;
      }

      try {
        const url = await colabService.getSessionUrl(sessionName);
        if (url) {
          vscode.env.openExternal(vscode.Uri.parse(url));
        } else {
          vscode.env.openExternal(vscode.Uri.parse('https://colab.research.google.com'));
        }
      } catch (err: unknown) {
        vscode.window.showErrorMessage(`Failed to get session URL: ${String(err)}`);
      }
    }),

    vscode.commands.registerCommand('colab.restartKernel', async (target?: unknown) => {
      const sessionName = resolveSessionName(target);
      if (!sessionName) {
        vscode.window.showWarningMessage('No Colab session selected.');
        return;
      }

      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Restarting kernel for session '${sessionName}'...`,
        },
        async () => {
          try {
            await colabService.restartKernel(sessionName);
            vscode.window.showInformationMessage(`Kernel restarted for '${sessionName}'.`);
          } catch (e: unknown) {
            vscode.window.showErrorMessage(`Failed to restart kernel: ${String(e)}`);
          }
        }
      );
    }),

    vscode.commands.registerCommand('colab.stopSession', async (target?: unknown) => {
      const sessionName = resolveSessionName(target);
      if (!sessionName) {
        vscode.window.showWarningMessage('No Colab session selected.');
        return;
      }

      const confirm = await vscode.window.showWarningMessage(
        `Are you sure you want to stop Colab session '${sessionName}'? The VM will be terminated.`,
        { modal: true },
        'Stop Session'
      );

      if (confirm !== 'Stop Session') {
        return;
      }

      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Stopping Colab session '${sessionName}'...`,
        },
        async () => {
          try {
            await colabService.stopSession(sessionName);
            vscode.window.showInformationMessage(`Colab session '${sessionName}' terminated.`);
          } catch (e: unknown) {
            vscode.window.showErrorMessage(`Failed to stop session: ${String(e)}`);
          }
        }
      );
    }),

    vscode.commands.registerCommand('colab.viewLogs', async (target?: unknown) => {
      const sessionName = resolveSessionName(target);
      if (!sessionName) {
        vscode.window.showWarningMessage('No Colab session selected.');
        return;
      }

      outputChannel.clear();
      outputChannel.show();
      outputChannel.appendLine(`=== Session Logs for '${sessionName}' ===\n`);

      try {
        const logs = await colabService.getLogs(sessionName);
        outputChannel.appendLine(logs);
      } catch (err: unknown) {
        outputChannel.appendLine(`Failed to retrieve logs: ${String(err)}`);
      }
    }),

    vscode.commands.registerCommand('colab.newSessionT4', async () => {
      const defaultName = `t4-gpu-${Date.now().toString().slice(-4)}`;
      const name = await vscode.window.showInputBox({
        prompt: 'Enter session name for new T4 GPU runtime',
        value: defaultName,
      });
      if (!name) {
        return;
      }

      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Provisioning T4 GPU session '${name}'...`,
        },
        async () => {
          try {
            await colabService.createSession(name, 'T4');
            vscode.window.showInformationMessage(`T4 GPU session '${name}' is READY.`);
          } catch (err: unknown) {
            vscode.window.showErrorMessage(`Failed to provision T4 GPU session: ${String(err)}`);
          }
        }
      );
    }),

    vscode.commands.registerCommand('colab.newSessionL4', async () => {
      const defaultName = `l4-gpu-${Date.now().toString().slice(-4)}`;
      const name = await vscode.window.showInputBox({
        prompt: 'Enter session name for new L4 GPU runtime',
        value: defaultName,
      });
      if (!name) {
        return;
      }

      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Provisioning L4 GPU session '${name}'...`,
        },
        async () => {
          try {
            await colabService.createSession(name, 'L4');
            vscode.window.showInformationMessage(`L4 GPU session '${name}' is READY.`);
          } catch (err: unknown) {
            vscode.window.showErrorMessage(`Failed to provision L4 GPU session: ${String(err)}`);
          }
        }
      );
    }),

    vscode.commands.registerCommand('colab.newSessionA100', async () => {
      const defaultName = `a100-gpu-${Date.now().toString().slice(-4)}`;
      const name = await vscode.window.showInputBox({
        prompt: 'Enter session name for new A100 High-RAM GPU runtime',
        value: defaultName,
      });
      if (!name) {
        return;
      }

      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Provisioning A100 High-RAM GPU session '${name}'...`,
        },
        async () => {
          try {
            await colabService.createSession(name, 'A100', true);
            vscode.window.showInformationMessage(`A100 High-RAM session '${name}' is READY.`);
          } catch (err: unknown) {
            vscode.window.showErrorMessage(`Failed to provision A100 GPU session: ${String(err)}`);
          }
        }
      );
    }),

    vscode.commands.registerCommand('colab.newSessionCustom', async () => {
      const acceleratorChoice = await vscode.window.showQuickPick(
        [
          { label: 'T4 GPU', description: '~1.5-2 CCU/hr (Standard 16GB)', accel: 'T4' as const },
          { label: 'L4 GPU', description: '~2-3 CCU/hr (Ada Lovelace 24GB)', accel: 'L4' as const },
          { label: 'A100 GPU', description: '~10-13 CCU/hr (High performance)', accel: 'A100' as const },
          { label: 'CPU Runtime', description: 'Standard compute units', accel: 'CPU' as const },
        ],
        { placeHolder: 'Select hardware accelerator variant' }
      );

      if (!acceleratorChoice) {
        return;
      }

      let highMem = false;
      if (acceleratorChoice.accel !== 'L4') {
        const memChoice = await vscode.window.showQuickPick(
          ['Standard RAM', 'High-RAM'],
          { placeHolder: 'Select machine memory shape' }
        );
        highMem = memChoice === 'High-RAM';
      }

      const defaultName = `session-${acceleratorChoice.accel.toLowerCase()}-${Date.now().toString().slice(-4)}`;
      const name = await vscode.window.showInputBox({
        prompt: 'Enter session name',
        value: defaultName,
      });

      if (!name) {
        return;
      }

      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Provisioning ${acceleratorChoice.label} '${name}'...`,
        },
        async () => {
          try {
            await colabService.createSession(name, acceleratorChoice.accel, highMem);
            vscode.window.showInformationMessage(`Colab session '${name}' is READY.`);
          } catch (err: unknown) {
            vscode.window.showErrorMessage(`Failed to provision session: ${String(err)}`);
          }
        }
      );
    })
  );

  // Initial update
  refreshAll(false);
}

export function deactivate() {
  clearInterval(autoRefreshTimer);
}
