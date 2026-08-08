import os
import pytest
from fastapi.testclient import TestClient

from web.app import app
from web.db import get_db, create_user, create_session, create_conversation, create_message

@pytest.fixture(autouse=True)
def clear_db():
    with get_db() as conn:
        conn.execute("DELETE FROM sessions;")
        conn.execute("DELETE FROM messages;")
        conn.execute("DELETE FROM conversations;")
        conn.execute("DELETE FROM users;")
        conn.execute("DELETE FROM guest_usage;")
        conn.commit()
    yield

def test_unauthenticated_me():
    client = TestClient(app)
    response = client.get("/api/auth/me")
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] is False
    assert data["mode"] == "guest"
    assert "rex_guest_session" in response.cookies

def test_guest_session_cookie_reuse():
    client = TestClient(app)
    res1 = client.get("/api/auth/me")
    guest_cookie = res1.cookies.get("rex_guest_session")
    assert guest_cookie is not None
    
    res2 = client.get("/api/auth/me", cookies={"rex_guest_session": guest_cookie})
    assert res2.json()["guest_session"] == guest_cookie

def test_authenticated_me():
    user_id = create_user("google-123", "Nafis", "nafis@gmail.com", "http://avatar.url")
    session_id = create_session(user_id)
    
    client = TestClient(app)
    response = client.get("/api/auth/me", cookies={"rex_user_session": session_id})
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] is True
    assert data["mode"] == "user"
    assert data["user"]["name"] == "Nafis"
    assert data["user"]["email"] == "nafis@gmail.com"

def test_google_oauth_route():
    client = TestClient(app)
    response = client.get("/api/auth/google", follow_redirects=False)
    assert response.status_code == 307
    assert "accounts.google.com" in response.headers["Location"]
    assert "google_oauth_state" in response.cookies

def test_logout():
    user_id = create_user("google-123", "Nafis", "nafis@gmail.com", "http://avatar.url")
    session_id = create_session(user_id)
    
    client = TestClient(app)
    res = client.post("/api/auth/logout", cookies={"rex_user_session": session_id})
    assert res.status_code == 200
    assert "rex_user_session" not in res.cookies or res.cookies.get("rex_user_session") == ""

def test_guest_usage_limit(monkeypatch):
    monkeypatch.setenv("GUEST_DAILY_LIMIT", "0")
    client = TestClient(app)
    
    res = client.post("/api/chat", json={"message": "Halo"}, cookies={"rex_guest_session": "guest-session-123"})
    assert res.status_code == 403
    assert "Guest limit" in res.json()["error"]
    
    # Authenticated user is exempt from guest limit
    user_id = create_user("google-123", "Nafis", "nafis@gmail.com", "http://avatar.url")
    session_id = create_session(user_id)
    
    res_auth = client.post("/api/chat", json={"message": "Halo"}, cookies={"rex_user_session": session_id})
    assert res_auth.status_code != 403

def test_chat_ownership():
    user_a = create_user("google-a", "User A", "a@gmail.com", "")
    session_a = create_session(user_a)
    conv_a = create_conversation(user_id=user_a, title="Chat A")
    
    user_b = create_user("google-b", "User B", "b@gmail.com", "")
    session_b = create_session(user_b)
    
    client = TestClient(app)
    res = client.get(f"/api/conversations/{conv_a}", cookies={"rex_user_session": session_b})
    assert res.status_code == 403
    
    res_guest = client.get(f"/api/conversations/{conv_a}", cookies={"rex_guest_session": "guest-id"})
    assert res_guest.status_code == 403

def test_conversation_migration():
    guest_session = "guest-session-uuid"
    conv_id = create_conversation(guest_session_id=guest_session, title="Guest Chat")
    create_message(conv_id, "user", "Hello from Guest")
    
    user_id = create_user("google-123", "Nafis", "nafis@gmail.com", "")
    
    from web.db import migrate_guest_chats, get_conversation
    migrate_guest_chats(guest_session, user_id)
    
    conv = get_conversation(conv_id)
    assert conv["user_id"] == user_id
