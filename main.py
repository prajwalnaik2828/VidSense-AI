"""VidSense AI - Desktop Application Launcher.

Runs the VidSense backend server and launches the application as an isolated
native desktop window (without browser tabs or URL bar) using the native Chromium/Edge
App Mode. Cleanly shuts down the server when the desktop application window is closed.
"""

import os
import sys
import time
import shutil
import signal
import subprocess
import requests

from vidsense import config


def find_app_browser() -> str | None:
    """Finds available Edge, Chrome, or Chromium binary supporting --app mode."""
    candidates = [
        # Microsoft Edge
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        # Google Chrome
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        # Brave Browser
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    # Check PATH
    for cmd in ["msedge", "chrome", "google-chrome", "chromium", "brave"]:
        found = shutil.which(cmd)
        if found:
            return found

    return None


def find_python_executable(root_dir: str) -> str:
    """Finds available virtual environment python executable or falls back to sys.executable."""
    candidates = [
        os.path.join(root_dir, ".venv", "Scripts", "python.exe"),
        os.path.join(root_dir, "venv", "Scripts", "python.exe"),
        os.path.join(root_dir, ".venv", "bin", "python"),
        os.path.join(root_dir, "venv", "bin", "python"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return sys.executable


def wait_for_server(
    base_url: str,
    proc: subprocess.Popen | None = None,
    log_path: str | None = None,
    timeout: int = 90,
) -> bool:
    """Waits for the backend server to respond with 200 OK on /health."""
    start = time.time()
    print("Initializing VidSense neural models and database", end="", flush=True)
    while time.time() - start < timeout:
        if proc and proc.poll() is not None:
            print(" [Server process exited unexpectedly]")
            if log_path and os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        tail = f.read().strip()
                        if tail:
                            print("\n--- Backend Server Log ---")
                            print(tail)
                except Exception:
                    pass
            return False
        try:
            r = requests.get(f"{base_url}/health", timeout=1.5)
            if r.status_code == 200:
                print(" [Ready]")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(1.0)
    print(" [Timeout]")
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.read().strip()
                if tail:
                    print("\n--- Backend Server Log (at timeout) ---")
                    print(tail[-1500:])
        except Exception:
            pass
    return False


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    profile_dir = os.path.join(root_dir, "data", "app_profile")
    os.makedirs(profile_dir, exist_ok=True)

    base_url = f"http://127.0.0.1:{config.API_PORT}"

    print("=" * 60)
    print("           VidSense AI - Vision Intelligence Core           ")
    print("=" * 60)
    print(f"Launching backend server on {base_url}...")

    # Start backend server in dedicated subprocess with logs routed to data/server.log
    log_path = os.path.join(root_dir, "data", "server.log")
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")

    python_exe = find_python_executable(root_dir)
    server_cmd = [python_exe, os.path.join(root_dir, "run_server.py")]
    server_proc = subprocess.Popen(
        server_cmd,
        cwd=root_dir,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    def cleanup():
        print("\nClosing VidSense AI application...")
        try:
            server_proc.terminate()
            server_proc.wait(timeout=4)
        except Exception:
            try:
                server_proc.kill()
            except Exception:
                pass
        try:
            log_file.close()
        except Exception:
            pass
        print("Backend server stopped.")

    try:
        # Wait for server readiness
        if not wait_for_server(base_url, proc=server_proc, log_path=log_path, timeout=90):
            print("Error: Backend server failed to start within timeout.")
            cleanup()
            sys.exit(1)

        browser_exe = find_app_browser()

        if browser_exe:
            print(f"Launching native desktop window via: {os.path.basename(browser_exe)}")
            app_args = [
                browser_exe,
                f"--app={base_url}/",
                f"--user-data-dir={profile_dir}",
                "--window-size=1380,880",
                "--app-auto-launched",
                "--no-first-run",
                "--no-default-browser-check",
            ]

            app_proc = subprocess.Popen(app_args)
            # Wait for user to close desktop window
            app_proc.wait()
        else:
            # Fallback to standard web browser
            import webbrowser
            print(f"Opening VidSense UI in default browser: {base_url}/")
            webbrowser.open(f"{base_url}/")
            print("Press Ctrl+C in this terminal to shut down the server.")
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
