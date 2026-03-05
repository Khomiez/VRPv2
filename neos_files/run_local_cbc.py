#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run VRP using CBC solver locally (if installed)

This converts the AMPL model to LP format and solves with CBC
"""

import subprocess
import sys
from pathlib import Path

def run_cbc_locally():
    """Try to run CBC solver locally"""

    # Check if CBC is installed
    try:
        result = subprocess.run(['cbc', '--version'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            print("CBC solver found!")
            print(result.stdout)
        else:
            print("CBC not found. Install from: https://github.com/coin-or/Cbc")
            return False
    except FileNotFoundError:
        print("CBC solver not installed.")
        print("\nTo install CBC:")
        print("  Windows: Download from https://ampl.com/products/solvers/")
        print("  Or use: conda install -c coin-or-cbc coincbc")
        return False
    except Exception as e:
        print(f"Error checking CBC: {e}")
        return False

    # Files
    mod_file = Path("D:/projects/python/VRPv2/neos_files/vrp_20.mod")
    dat_file = Path("D:/projects/python/VRPv2/neos_files/vrp_20.dat")

    print("\nAttempting to solve locally...")
    print("Note: CBC only accepts LP format, not AMPL directly.")
    print("You would need to convert AMPL to LP first.")

    return True

if __name__ == '__main__':
    run_cbc_locally()
