#!/usr/bin/env python3
import json
import os
import sqlite3
import sys

# Ensure UTF-8 output on Windows consoles (Python 3.7+).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def check_file_exists(filepath, required=True):
    if not os.path.exists(filepath):
        if required:
            print(f"[X] MISSING: {filepath}")
            return False
        print(f"[i] OPTIONAL MISSING: {filepath}")
        return True
    print(f"[OK] FOUND: {filepath}")
    return True

def verify_json(filepath, required=True):
    if not check_file_exists(filepath, required=required):
        return False
    if not os.path.exists(filepath):
        return True
    try:
        with open(filepath, encoding="utf-8") as f:
            json.load(f)
        print(f"[OK] VALID JSON: {filepath}")
        return True
    except Exception as e:
        print(f"[X] INVALID JSON: {filepath} - {str(e)}")
        return False

def verify_sqlite(filepath, required=True):
    if not check_file_exists(filepath, required=required):
        return False
    if not os.path.exists(filepath):
        return True
    try:
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        print(f"[OK] VALID SQLITE: {filepath} ({len(tables)} tables found)")
        return True
    except Exception as e:
        print(f"[X] INVALID SQLITE: {filepath} - {str(e)}")
        return False

def verify_env(filepath):
    if not check_file_exists(filepath):
        return False
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
        keys = [line.split('=')[0].strip() for line in lines if '=' in line and not line.startswith('#')]
        print(f"[OK] VALID ENV: {filepath} ({len(keys)} keys found)")
        return True
    except Exception as e:
        print(f"[X] INVALID ENV: {filepath} - {str(e)}")
        return False

def main():
    user_data_dir = os.path.join(os.getcwd(), "user_data")
    if not os.path.exists(user_data_dir):
        print(f"[!] user_data directory not found at {user_data_dir}")
        sys.exit(0)  # Not an error if fresh install

    results = []
    results.append(verify_json(os.path.join(user_data_dir, "chat_history.json")))
    results.append(verify_json(os.path.join(user_data_dir, "knowledge_graph.json")))
    results.append(verify_json(os.path.join(user_data_dir, "user_memory.json")))
    results.append(verify_sqlite(os.path.join(user_data_dir, "checkpoints.sqlite"), required=False))
    results.append(verify_env(os.path.join(user_data_dir, ".env")))

    if all(results):
        print("\n[OK] Data integrity check PASSED")
        sys.exit(0)
    else:
        print("\n[X] Data integrity check FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
