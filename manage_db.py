"""
manage_db.py — View and manage the face recognition database.

Usage:
    python manage_db.py list
    python manage_db.py delete --name "John"
    python manage_db.py reset
"""

import argparse
from database import init_db, list_users, delete_user, get_connection

parser = argparse.ArgumentParser(description="Manage the face recognition database.")
sub = parser.add_subparsers(dest="command")

sub.add_parser("list",  help="List all registered users")

del_p = sub.add_parser("delete", help="Delete a user by name")
del_p.add_argument("--name", required=True, help="Name to delete")

sub.add_parser("reset", help="⚠️  Delete ALL users from the database")

args = parser.parse_args()

init_db()

if args.command == "list":
    list_users()

elif args.command == "delete":
    confirm = input(f"Delete all records for '{args.name}'? [y/N]: ").strip().lower()
    if confirm == 'y':
        delete_user(args.name)
    else:
        print("Cancelled.")

elif args.command == "reset":
    confirm = input("⚠️  This will DELETE ALL users. Type 'yes' to confirm: ").strip()
    if confirm == 'yes':
        with get_connection() as conn:
            conn.execute("DELETE FROM users")
            conn.commit()
        print("✅ All users deleted.")
    else:
        print("Cancelled.")

else:
    parser.print_help()
