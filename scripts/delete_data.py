"""
delete_data.py — Delete the entire data/ folder.

This script permanently deletes the data/ directory and all its contents.

Usage:
    python scripts/delete_data.py              # Interactive with confirmation
    python scripts/delete_data.py --force      # Skip confirmation
    python scripts/delete_data.py --dry-run    # Preview without deleting
"""

import os
import sys
import shutil
import argparse

def get_project_root():
    """Return project root (parent of scripts/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_dir_size(path):
    """Calculate directory size in bytes."""
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    except:
        pass
    return total

def format_size(bytes):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} TB"

def main():
    parser = argparse.ArgumentParser(description="Delete data/ folder")
    parser.add_argument("--force", action="store_true", help="Skip confirmation")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()
    
    project_root = get_project_root()
    data_path = os.path.join(project_root, "data")
    
    print("=" * 60)
    print("Delete Data Folder")
    print("=" * 60)
    print(f"Target: {data_path}")
    
    if not os.path.exists(data_path):
        print("\n✓ data/ folder does not exist. Nothing to delete.")
        return
    
    size = get_dir_size(data_path)
    print(f"Size  : {format_size(size)}")
    print("=" * 60)
    
    if args.dry_run:
        print("\n[DRY RUN] Would delete the entire data/ folder")
        return
    
    if not args.force:
        response = input("\nPermanently delete data/ folder? [y/N]: ").strip().lower()
        if response not in ('y', 'yes'):
            print("Cancelled.")
            return
    
    print("\nDeleting data/ folder...")
    try:
        shutil.rmtree(data_path)
        print("✓ Successfully deleted data/ folder")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
