#!/usr/bin/env python3
"""
Colab Experiment Runner Helper Script:
Automates syncing the local repository to Google Colab, executing benchmark/experiment
scripts on remote GPU hardware (T4, L4, A100), and pulling result artifacts back locally.

Usage Examples:
    # Run a standalone script on an ephemeral T4 GPU:
    python3 scripts/colab_runner.py --ephemeral --gpu T4 benchmarks/benchmark_test.py

    # Sync workspace and run hardware scaling benchmark on Colab GPU:
    python3 scripts/colab_runner.py --sync --gpu T4 benchmarks/run_hardware_scaling.py -- --device cuda

    # Run on high-RAM A100 GPU and pull output JSON/plots:
    python3 scripts/colab_runner.py --sync --gpu A100 --high-mem --pull hardware_scaling_results.json benchmarks/run_hardware_scaling.py -- --device cuda
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def find_colab_cli() -> str:
    colab = shutil.which("colab")
    if colab:
        return colab
    default_path = os.path.expanduser("~/.local/bin/colab")
    if os.path.exists(default_path):
        return default_path
    raise RuntimeError("Google Colab CLI ('colab') not found in PATH or ~/.local/bin/colab.")


def run_ephemeral(script_path: str, gpu: str, high_mem: bool, script_args: list[str]) -> int:
    colab = find_colab_cli()
    cmd = [colab, "run"]
    if gpu and gpu.upper() != "CPU":
        cmd.extend(["--gpu", gpu.upper()])
    if high_mem:
        cmd.append("--high-mem")
    cmd.append(script_path)
    if script_args:
        cmd.extend(script_args)

    print(f"[Colab Runner] Launching ephemeral VM ({gpu})...")
    print(f"[Colab Runner] Command: {' '.join(cmd)}")
    return subprocess.call(cmd)


def run_synced_session(
    script_path: str,
    session_name: str,
    gpu: str,
    high_mem: bool,
    pull_patterns: list[str],
    script_args: list[str],
) -> int:
    colab = find_colab_cli()
    project_name = REPO_ROOT.name
    remote_dir = f"/content/{project_name}"

    # 1. Check or start session
    print(f"[Colab Runner] Checking active session '{session_name}'...")
    check = subprocess.run([colab, "status", "-s", session_name], capture_output=True, text=True)
    if check.returncode != 0 or f"Session '{session_name}' not found" in check.stdout:
        print(f"[Colab Runner] Provisioning new Colab session '{session_name}' ({gpu})...")
        new_cmd = [colab, "new", "-s", session_name]
        if gpu and gpu.upper() != "CPU":
            new_cmd.extend(["--gpu", gpu.upper()])
        if high_mem:
            new_cmd.append("--high-mem")
        subprocess.check_call(new_cmd)

    # 2. Sync workspace via rsync
    print(f"[Colab Runner] Syncing {REPO_ROOT} -> {session_name}:{remote_dir}...")
    rsync_cmd = [
        "rsync",
        "-avz",
        "--delete",
        "--exclude", ".git",
        "--exclude", ".venv",
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", ".pytest_cache",
        "-e", f"{colab} ssh --proxy-mode -s {session_name} -o StrictHostKeyChecking=no",
        f"{REPO_ROOT}/",
        f"root@colab-runtime:{remote_dir}/",
    ]
    subprocess.check_call(rsync_cmd)

    # 3. Execute script remotely
    rel_script = Path(script_path).resolve().relative_to(REPO_ROOT)
    args_str = " ".join(script_args)
    exec_cmd_str = f"cd {remote_dir} && python3 {rel_script} {args_str}"
    print(f"[Colab Runner] Executing remotely: {exec_cmd_str}")

    ssh_cmd = [
        "ssh",
        "-o", f"ProxyCommand={colab} ssh --proxy-mode -s {session_name}",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-t",
        "root@colab-runtime",
        exec_cmd_str,
    ]
    exit_code = subprocess.call(ssh_cmd)

    # 4. Pull requested artifacts
    if pull_patterns:
        for pattern in pull_patterns:
            print(f"[Colab Runner] Pulling artifact: {pattern}...")
            pull_cmd = [
                "scp",
                "-o", f"ProxyCommand={colab} ssh --proxy-mode -s {session_name}",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                f"root@colab-runtime:{remote_dir}/{pattern}",
                f"{REPO_ROOT}/",
            ]
            try:
                subprocess.call(pull_cmd)
            except Exception as e:
                print(f"[Colab Runner] Warning: could not pull '{pattern}': {e}")

    return exit_code


def main():
    parser = argparse.ArgumentParser(description="Run numerical experiments on Google Colab GPU hardware.")
    parser.add_argument("script", help="Path to Python script to execute")
    parser.add_argument("--gpu", default="T4", choices=["T4", "L4", "A100", "CPU"], help="GPU accelerator variant")
    parser.add_argument("--high-mem", action="store_true", help="Request high-memory shape")
    parser.add_argument("--session", default="colab-gpu", help="Name of persistent session")
    parser.add_argument("--ephemeral", action="store_true", help="Use one-shot ephemeral VM (colab run)")
    parser.add_argument("--sync", action="store_true", help="Sync local repo to Colab before execution")
    parser.add_argument("--pull", action="append", default=[], help="Pattern or relative path to pull back locally after run")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the script")

    args = parser.parse_args()

    # Clean script args (remove leading '--' if present)
    script_args = args.script_args
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    if args.ephemeral:
        sys.exit(run_ephemeral(args.script, args.gpu, args.high_mem, script_args))
    else:
        sys.exit(
            run_synced_session(
                args.script,
                args.session,
                args.gpu,
                args.high_mem,
                args.pull,
                script_args,
            )
        )


if __name__ == "__main__":
    main()
