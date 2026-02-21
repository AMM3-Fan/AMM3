#!/usr/bin/env python3
"""
Environment.py - System mapping tool for AMM3.

Provides situational awareness of the host environment (OS, architecture,
installed software, shell) to improve model command accuracy.
"""

import os
import subprocess
import sys
import platform


def get_system_report() -> str:
    """Generate a concise one-line summary of the host environment."""
    os_name = "macOS" if sys.platform == "darwin" else sys.platform
    arch = platform.machine()
    shell = os.path.basename(os.environ.get("SHELL", "unknown"))

    # OS version detection with explicit error handling
    if sys.platform == "darwin":
        try:
            os_version = subprocess.check_output(
                ["sw_vers", "-productVersion"], text=True, timeout=2
            ).strip()
        except subprocess.TimeoutExpired:
            os_version = "unknown (sw_vers timed out)"
        except subprocess.CalledProcessError:
            os_version = "unknown (sw_vers failed)"
        except FileNotFoundError:
            os_version = "unknown (sw_vers not found)"
    else:
        os_version = platform.release()

    return f"OS: {os_name} {os_version}, Arch: {arch}, Shell: {shell}"


if __name__ == "__main__":
    print(get_system_report())
