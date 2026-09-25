#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated synchronization utility between local project repository
and Google Drive mounted folder.

Target Google Drive location:
GEMINI-APPLICATIONS/mofs-her-oer-co2rr
"""

import os
import sys
import shutil
from pathlib import Path

# Paths
LOCAL_REPO = Path(__file__).resolve().parent
GDRIVE_DEST = Path("/run/user/1000/gvfs/google-drive:host=gmail.com,user=ribeirojr.fis/0AJMdF0C-r4sVUk9PVA/1uQiXpQIq6Eql-ILNBtF1Y8bYh1iKT6M_/1_OXiyrxZK0tQM0gs1WP6S4hOg-XBixfz")

# Exclude patterns
EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", ".vscode", "tmp"}
EXCLUDE_EXTS = {".pyc", ".pyo", ".swp"}

def sync_directory(src_dir: Path, dst_dir: Path):
    if not dst_dir.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
    
    synced_count = 0
    for root, dirs, files in os.walk(src_dir):
        # Modify dirs in-place to avoid descending into excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        rel_path = Path(root).relative_to(src_dir)
        target_dir = dst_dir / rel_path
        
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            
        for file in files:
            if any(file.endswith(ext) for ext in EXCLUDE_EXTS):
                continue
            src_file = Path(root) / file
            dst_file = target_dir / file
            
            # Check if copy is needed based on size or existence
            need_copy = False
            if not dst_file.exists():
                need_copy = True
            else:
                try:
                    if src_file.stat().st_size != dst_file.stat().st_size:
                        need_copy = True
                except Exception:
                    need_copy = True
            
            if need_copy:
                try:
                    shutil.copyfile(src_file, dst_file)
                    synced_count += 1
                except Exception as e:
                    print(f"[WARN] Failed to copy {src_file.name}: {e}")

    print(f"[SYNC] Successfully mirrored repository to Google Drive ({synced_count} files updated).")

if __name__ == "__main__":
    if not GDRIVE_DEST.exists():
        print(f"[ERROR] Google Drive mount not found at {GDRIVE_DEST}")
        sys.exit(1)
    print(f"[SYNC] Starting sync from {LOCAL_REPO} -> {GDRIVE_DEST}")
    sync_directory(LOCAL_REPO, GDRIVE_DEST)
