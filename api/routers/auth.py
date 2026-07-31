"""
Auth endpoints — login / logout / whoami.

A successful login returns a Bearer access token (for the iOS app) and also
sets an httponly ``cairniq_token`` cookie plus the legacy ``profile`` cookie so
the existing browser UI keeps working. The profile middleware in ``server.py``
binds the active profile from whichever the request presents.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tools.auth import (
    auth_required,
    create_user,
    extract_bearer,
    issue_token,
    list_users,
    verify_credentials,
    verify_token,
)

router = APIRouter()

# Cookie carrying the JWT for browser clients. httponly so page JS can't read
# it; samesite=lax is fine because the app posts same-origin.
TOKEN_COOKIE = "cairniq_token"


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


def _extract_token(request: Request) -> str | None:
    """Pull a bearer token from the Authorization header or the cookie."""
    return extract_bearer(
        request.headers.get("authorization"), request.cookies.get(TOKEN_COOKIE)
    )


@router.post("/api/auth/register")
async def register(req: RegisterRequest):
    try:
        # If no users exist, the first user is an admin
        existing_users = list_users()
        role = "admin" if not existing_users else "user"

        user = create_user(
            username=req.username,
            password=req.password,
            profile=None,
            role=role,
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Issue token and log the user in immediately
    token, expires_in = issue_token(user)
    response = JSONResponse(
        {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": expires_in,
            "username": user["username"],
            "profile": user["profile"],
            "role": user.get("role", "user"),
        }
    )
    response.set_cookie(
        key=TOKEN_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=expires_in,
    )
    response.set_cookie(
        key="profile",
        value=user["profile"],
        httponly=True,
        samesite="lax",
        max_age=expires_in,
    )
    return response


@router.post("/api/auth/login")
async def login(req: LoginRequest):
    user = verify_credentials(req.username, req.password)
    if user is None:
        # Generic message — never reveal whether the username exists.
        return JSONResponse({"error": "Invalid username or password."}, status_code=401)

    token, expires_in = issue_token(user)
    response = JSONResponse(
        {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": expires_in,
            "username": user["username"],
            "profile": user["profile"],
            "role": user.get("role", "user"),
        }
    )
    response.set_cookie(
        key=TOKEN_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=expires_in,
    )
    # Keep the legacy profile cookie in sync so existing browser flows resolve
    # the same profile even before auth enforcement is switched on.
    response.set_cookie(
        key="profile",
        value=user["profile"],
        httponly=True,
        samesite="lax",
        max_age=expires_in,
    )
    return response


@router.post("/api/auth/logout")
async def logout():
    response = JSONResponse({"status": "logged_out"})
    response.delete_cookie(TOKEN_COOKIE)
    response.delete_cookie("profile")
    return response


@router.get("/api/auth/me")
async def me(request: Request):
    claims = verify_token(_extract_token(request) or "")
    if claims is None:
        return JSONResponse({"error": "Not authenticated."}, status_code=401)
    return JSONResponse(
        {
            "username": claims.get("sub"),
            "profile": claims.get("profile"),
            "role": claims.get("role", "user"),
            "auth_required": auth_required(),
        }
    )
