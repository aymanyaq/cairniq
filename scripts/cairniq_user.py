#!/usr/bin/env python3
"""
CairnIQ household user administration.

Manage the login accounts that the auth layer maps to profiles. Run from the
project root:

    python scripts/cairniq_user.py add alice --profile alice --role admin
    python scripts/cairniq_user.py list
    python scripts/cairniq_user.py passwd alice
    python scripts/cairniq_user.py remove olduser

Passwords are read interactively (never from argv) unless --password is given.
The store lives at user_data/auth.json (gitignored); override with CAIRNIQ_AUTH_DB.
"""

import argparse
import getpass
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.auth import (  # noqa: E402
    create_user,
    delete_user,
    list_users,
    set_password,
)


def _prompt_password(confirm: bool = True) -> str:
    pw = getpass.getpass("Password: ")
    if confirm:
        again = getpass.getpass("Confirm password: ")
        if pw != again:
            print("Passwords do not match.", file=sys.stderr)
            sys.exit(1)
    return pw


def cmd_add(args: argparse.Namespace) -> int:
    password = args.password or _prompt_password()
    try:
        user = create_user(
            args.username, password, profile=args.profile, role=args.role
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Created user '{user['username']}' -> profile '{user['profile']}' ({user['role']}).")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    users = list_users()
    if not users:
        print("No users yet. Add one with: python scripts/cairniq_user.py add <name>")
        return 0
    width = max(len(u["username"]) for u in users)
    for u in users:
        print(f"  {u['username']:<{width}}  profile={u['profile']:<16} role={u['role']}")
    return 0


def cmd_passwd(args: argparse.Namespace) -> int:
    password = args.password or _prompt_password()
    try:
        ok = set_password(args.username, password)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if not ok:
        print(f"User '{args.username}' not found.", file=sys.stderr)
        return 1
    print(f"Password updated for '{args.username}'. Existing tokens are now invalid.")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    if delete_user(args.username):
        print(f"Removed user '{args.username}'.")
        return 0
    print(f"User '{args.username}' not found.", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage CairnIQ login accounts.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Create a new user")
    p_add.add_argument("username")
    p_add.add_argument("--profile", help="Profile to bind (defaults to username)")
    p_add.add_argument("--role", choices=["user", "admin"], default="user")
    p_add.add_argument("--password", help="Set non-interactively (avoid in shared shells)")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="List users")
    p_list.set_defaults(func=cmd_list)

    p_pw = sub.add_parser("passwd", help="Change a user's password")
    p_pw.add_argument("username")
    p_pw.add_argument("--password")
    p_pw.set_defaults(func=cmd_passwd)

    p_rm = sub.add_parser("remove", help="Delete a user")
    p_rm.add_argument("username")
    p_rm.set_defaults(func=cmd_remove)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
