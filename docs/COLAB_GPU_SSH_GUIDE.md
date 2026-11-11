# Google Colab Pro GPU SSH & Remote Execution Guide

This guide details how to connect directly to Google Colab GPU runtimes (T4, L4, A100, H100) via SSH, OpenSSH configuration, and remote CLI execution from your local machine.

---

## 1. Fast Setup: Colab CLI (Official Google Tool)

We have installed the official Google Colab CLI (`colab`, located at `~/.local/bin/colab`) and configured your local SSH client.

### Step 1: One-Time Google Account Authentication

Because you have a Colab Pro subscription, connect your account by running:

```bash
colab sessions
```

1. It will output a Google OAuth URL in your terminal.
2. Open that URL in your browser and sign in with your Google account that has the Colab Pro subscription.
3. Grant access and copy the authorization code displayed on the landing page.
4. Paste the authorization code back into the terminal prompt.

The credentials are saved locally at `~/.config/colab-cli/token.json` and automatically refreshed.

---

## 2. Connecting via SSH

Your SSH configuration (`~/.ssh/config`) has been configured with your existing `~/.ssh/id_ed25519` key.

### Quick Connect Commands

| Command | GPU Accelerator | Description |
| :--- | :--- | :--- |
| `ssh colab-gpu` | **NVIDIA T4** (16GB) | Ideal for general testing, low compute unit cost (~1.5–2 units/hr) |
| `ssh colab-l4` | **NVIDIA L4** (24GB) | Modern Ada Lovelace GPU, high memory bandwidth (~2–3 units/hr) |
| `ssh colab-a100` | **NVIDIA A100** (40GB/80GB) + High-RAM | Maximum compute power for heavy benchmarks (~10–13 units/hr) |
| `ssh colab` | Existing session | Attaches to any currently active session without provisioning a new one |

When you run `ssh colab-gpu` (or any of the hosts above), the CLI:
1. Automatically provisions the requested GPU runtime on Google Colab if not already running.
2. Sends your public SSH key (`~/.ssh/id_ed25519.pub`) securely over the WebSocket handshake to root's `authorized_keys`.
3. Opens a full interactive PTY bash shell directly in `/content`.

### Using with VS Code Remote - SSH

You can develop directly on the Colab GPU instance using VS Code:
1. In VS Code, press `Ctrl+Shift+P` -> **Remote-SSH: Connect to Host...**
2. Select `colab-gpu` (or `colab-a100`).
3. VS Code will connect, install the VS Code Server inside `/content`, and give you the full IDE experience (terminal, file explorer, GPU debugger, extensions) on the remote Colab GPU.

---

## 3. Remote Execution & Benchmarking (`colab run` & `colab exec`)

You don't always need an interactive SSH shell to run benchmarks. The CLI supports headless execution:

### Ephemeral One-Shot Jobs (`colab run`)
Automatically rents the GPU, executes your script, streams output, and terminates the VM upon completion to save compute units:

```bash
# Run a benchmark script on an A100 GPU:
colab run --gpu A100 run_kan_experiments.py

# Run on a T4 GPU:
colab run --gpu T4 benchmarks/benchmark_sweep.py --batch-size 128
```

### Shebang Support
You can put this at the top of any standalone benchmark script:
```python
#!/usr/bin/env -S colab run --gpu A100 --keep
import torch

print("GPU:", torch.cuda.get_device_name(0))
```
Make it executable (`chmod +x script.py`) and simply execute `./script.py`.

### Piping Commands
```bash
# Check GPU hardware status:
echo "import torch; print('CUDA Available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0))" | colab exec -s colab-gpu

# Run nvidia-smi:
echo "nvidia-smi" | colab console -s colab-gpu
```

---

## 4. File Syncing (Local <-> Colab)

### Using `rsync` or `scp` over SSH
Since `colab-gpu` is in your `~/.ssh/config`, standard tools work seamlessly:

```bash
# Sync local repo to Colab /content:
rsync -avz --exclude '.git' --exclude '.venv' ./ colab-gpu:/content/lsing_sw/

# Copy results/plots back to local machine:
scp -r colab-gpu:/content/lsing_sw/experiments/results/ ./experiments/results/
```

### Using Colab CLI built-ins:
```bash
colab upload -s colab-gpu ./requirements.txt /content/requirements.txt
colab download -s colab-gpu /content/benchmark_results.csv ./benchmark_results.csv
```

---

## 5. Session Management & Compute Units

Google Colab Pro uses a compute unit balance. Always remember to monitor and stop idle runtimes:

```bash
# Check remaining compute unit balance and burn rate:
colab usage

# List all active sessions:
colab sessions

# Stop a session to avoid consuming credits:
colab stop -s colab-gpu
colab stop -s colab-a100
```

---

## 6. Alternative: Tailscale SSH (Direct VPN to Colab)

If you start a session through the Google Colab web UI (https://colab.research.google.com), you can also join the VM directly to your local Tailscale tailnet:

In a Colab notebook cell:
```python
# 1. Install Tailscale
!curl -fsSL https://tailscale.com/install.sh | sh

# 2. Start tailscaled in userspace mode (required for Colab container permissions)
!nohup tailscaled --tun=userspace-networking > /dev/null 2>&1 &
import time; time.sleep(3)

# 3. Connect to your tailnet (create an ephemeral auth key at login.tailscale.com/admin/settings/keys)
!tailscale up --authkey=tskey-auth-YOUR_EPHEMERAL_KEY --ssh --hostname=colab-notebook

# 4. Show IP
!tailscale ip -4
```

Then from your local terminal (which is already connected to Tailscale):
```bash
tailscale ssh root@colab-notebook
```
