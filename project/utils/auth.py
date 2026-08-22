"""
Basic username/password authentication with role assignment (doctor vs
patient). Credentials live in config/users.json, with passwords stored as
SHA-256 hashes rather than plaintext.

Note on security level: this is "basic login" as requested -- hashing
without salting, no rate limiting, no session expiry, no account lockout.
It's a reasonable step up from a plain role dropdown for a thesis demo /
internal tool, but it is NOT a production-grade auth system and shouldn't
be trusted with real patient data as-is.
"""

import hashlib
import json
from pathlib import Path


def hash_password(password: str) -> str:
    """
    Hash a plaintext password with SHA-256.

    Used both to check a login attempt and to generate new entries for
    config/users.json when adding a user. To add a user, run:
        python -c "from utils.auth import hash_password; print(hash_password('yourpassword'))"
    and paste the resulting hash into config/users.json.
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_users(config_path: Path) -> dict:
    """Load the users config file. Returns {} if it doesn't exist yet."""
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def verify_login(username: str, password: str, users: dict):
    """
    Check a username/password pair against the loaded users config.
    Returns the role string ("doctor" or "patient") on success, or None
    if the username doesn't exist or the password doesn't match.
    """
    user = users.get(username)
    if user is None:
        return None
    if hash_password(password) == user.get("password_hash"):
        return user.get("role")
    return None
