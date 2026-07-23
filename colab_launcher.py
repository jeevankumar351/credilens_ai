"""
CrediLens AI - Google Colab Launcher
=======================================
Helper script to launch the Streamlit app from inside Google Colab,
since Colab cannot open localhost ports directly. Uses `localtunnel`
(no signup/API key required) to expose the app publicly.

Run this AFTER model_training.py has completed, inside a Colab cell:

    !python colab_launcher.py

Or follow the manual steps in README.md.
"""

import subprocess
import time
import sys
import urllib.request


def run_in_colab():
    print("🚀 Launching CrediLens AI in Google Colab...\n")

    # 1. Install localtunnel (Node-based, ships with Colab's Node runtime)
    print("📦 Installing localtunnel...")
    subprocess.run(["npm", "install", "-g", "localtunnel"], check=False)

    # 2. Get the public IP that localtunnel will ask you to confirm
    try:
        ip = urllib.request.urlopen("https://ipv4.icanhazip.com").read().decode().strip()
        print(f"\n🔑 Your tunnel password (paste into the LT page if asked): {ip}\n")
    except Exception:
        print("⚠️ Could not fetch tunnel password IP — check https://ipv4.icanhazip.com manually.")

    # 3. Launch Streamlit in the background
    print("▶️  Starting Streamlit server...")
    streamlit_proc = subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]
    )
    time.sleep(8)

    # 4. Open the tunnel
    print("🌐 Opening public tunnel via localtunnel...")
    subprocess.run(["npx", "localtunnel", "--port", "8501"])

    streamlit_proc.terminate()


if __name__ == "__main__":
    run_in_colab()
