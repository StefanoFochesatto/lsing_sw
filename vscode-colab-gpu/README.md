# Google Colab GPU & Sessions Monitor (VS Code Extension)

A lightweight, dedicated VS Code extension that connects directly to the Google Colab CLI to monitor and control your Google Colab Pro GPU sessions, compute units balance, and remote SSH tunnels.

---

## Features

### 1. Activity Bar Sidebar ("Google Colab")
- **Active Sessions View**:
  - Displays all running sessions with hardware badges (T4, L4, A100, CPU), memory shape (Standard vs High-RAM), and execution state (IDLE / BUSY).
  - Expand each session to view endpoint URLs, kernel IDs, and last execution details.
  - **Inline Action Buttons**:
    - `$(terminal)` **SSH Terminal**: Launches an integrated terminal connecting straight into the session.
    - `$(globe)` **Colab Web**: Opens the notebook session in your browser.
    - `$(debug-restart)` **Restart Kernel**: Restarts the kernel remotely without dropping the VM.
    - `$(debug-stop)` **Stop VM**: Prompts confirmation and terminates the session to save compute units.
    - `$(output)` **View Logs**: Dumps structured session event logs into the Colab output channel.

- **Compute Units & Usage View**:
  - Live CCU balance display.
  - Hourly consumption burn rate (`CCU/hr`).
  - Active VM instance count.
  - Estimated remaining runtime at the current consumption rate.

- **Quick Actions View**:
  - 1-click launch buttons for T4, L4, and A100 High-RAM GPU runtimes.
  - Custom session wizard.
  - Open Visual Dashboard.
  - Authenticate Colab CLI.

### 2. Interactive Visual Dashboard
Run `Colab: Open Visual Colab Dashboard` (`colab.openDashboard`):
- Full-page graphical dashboard styled with VS Code theme colors.
- CCU gauge, live burn rate warnings, active session cards, and quick hardware provisioning.

### 3. Real-Time Status Bar
- Shows in the bottom right: `$(flame) Colab: 1 active (T4) | 48.5 CCU` (or `$(server-process) Colab: Idle | 48.5 CCU`).
- Alerts when units are burning. Clicking it opens the Colab Dashboard.

### 4. Zero-Delay State Synchronization
- Automatically watches `~/.config/colab-cli/sessions.json` so terminal commands (`colab new`, `colab stop`, `colab run`) immediately reflect in the VS Code sidebar.
- Background polling every 20 seconds (configurable via `colab.autoRefreshInterval`).

---

## Configuration Settings

| Setting | Default | Description |
| :--- | :--- | :--- |
| `colab.cliPath` | `/home/stefano/.local/bin/colab` | Path to the `colab` executable |
| `colab.autoRefreshInterval` | `20` | Interval in seconds for background polling (0 disables) |
| `colab.showStatusBarItem` | `true` | Show/hide the status bar item |

---

## Development & Recompiling

```bash
cd /home/stefano/lsing_sw/vscode-colab-gpu
npm run compile
```

The extension is symlinked into `~/.vscode/extensions/stefano.colab-gpu-monitor-0.1.0`.
