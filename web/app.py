import os
import torch
import uuid
import secrets
import urllib.parse
import httpx
from fastapi import FastAPI, HTTPException, Cookie, Response, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.tokenizer.simple_tokenizer import SimpleWordTokenizer
from src.model import DecoderOnlyTransformer
from src.inference import TextGenerator

app = FastAPI(title="REXA Multi-Model Chat Interface")

# Global variables for model, tokenizer, and generator
model = None
tokenizer = None
generator = None
active_model_version = "V4"
local_model_loaded = False

CHECKPOINT_PATH_V5 = "checkpoints/rexa_v5_instruction.pt"
VOCAB_PATH_V5 = "data/tokenizer/vocab_v5.json"

CHECKPOINT_PATH_V4 = "checkpoints/v4_best_model.pt"
VOCAB_PATH_V4 = "data/tokenizer/vocab_v4.json"

SYSTEM_INSTRUCTION = """Kamu adalah REXA, asisten AI cerdas, natural, helpful, dan conversational yang dikembangkan oleh REXSHIN.
Aturan perilaku kamu:
1. Jangan pernah mengaku sebagai Gemini atau Google. Selalu perkenalkan dirimu sebagai REXA.
2. Jawablah secara langsung, cerdas, dan jujur. Jika tidak tahu atau tidak yakin, katakan bahwa kamu tidak tahu daripada mengarang informasi.
3. Gunakan bahasa yang natural dan santai jika pengguna berbicara dengan santai (menggunakan bahasa Indonesia sehari-hari, slang, atau singkatan/typo). Jika pengguna berbicara formal, gunakan bahasa yang formal pula.
4. Gunakan Markdown untuk format teks agar mudah dibaca (gunakan tebal, list, atau blok kode jika diperlukan).
5. Mampu menjelaskan konsep rumit dengan sederhana (seperti menjelaskan ke anak SMP) atau memberikan detail teknis mendalam sesuai kebutuhan pengguna."""

@app.on_event("startup")
def startup_event():
    global model, tokenizer, generator, active_model_version, local_model_loaded
    
    # Initialize SQLite database
    from web.db import init_db
    try:
        init_db()
        print("REXA Database: Initialized successfully.")
    except Exception as e:
        print(f"REXA Database: Failed to initialize: {str(e)}")
    
    # Check if V5 is available, otherwise fall back to V4
    if os.path.exists(CHECKPOINT_PATH_V5) and os.path.exists(VOCAB_PATH_V5):
        checkpoint_path = CHECKPOINT_PATH_V5
        vocab_path = VOCAB_PATH_V5
        from src.tokenizer.simple_tokenizer import SimpleWordTokenizerV5
        tokenizer_class = SimpleWordTokenizerV5
        active_model_version = "V5"
    else:
        checkpoint_path = CHECKPOINT_PATH_V4
        vocab_path = VOCAB_PATH_V4
        tokenizer_class = SimpleWordTokenizer
        active_model_version = "V4"
        
    if not os.path.exists(checkpoint_path) or not os.path.exists(vocab_path):
        print("REXA Local: Checkpoint or vocabulary not found. Local model disabled.")
        local_model_loaded = False
        return
        
    try:
        tokenizer = tokenizer_class(lowercase=True)
        tokenizer.load_vocab(vocab_path)
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = DecoderOnlyTransformer(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        generator = TextGenerator(model, tokenizer, device="cpu")
        local_model_loaded = True
        print(f"REXA Local: Model {active_model_version} loaded successfully!")
    except Exception as e:
        print(f"REXA Local: Failed to load local model: {str(e)}")
        local_model_loaded = False

# --- Pydantic Schemas ---

class MessageItem(BaseModel):
    role: Optional[str] = "user"
    content: Optional[str] = ""

    def __init__(self, **data):
        if "sender" in data and "role" not in data:
            data["role"] = data["sender"]
        if "text" in data and "content" not in data:
            data["content"] = data["text"]
        super().__init__(**data)

class ChatRequest(BaseModel):
    message: Optional[str] = None
    messages: Optional[List[MessageItem]] = None
    model: Optional[str] = "local"
    history: Optional[List[MessageItem]] = None
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 10
    max_new_tokens: Optional[int] = 50
    conversation_id: Optional[str] = None

# --- Authentication Endpoints ---

@app.get("/api/auth/me")
@app.get("/api/auth/status")
def auth_me(
    response: Response,
    rex_user_session: Optional[str] = Cookie(None),
    rex_guest_session: Optional[str] = Cookie(None)
):
    guest_id = rex_guest_session
    if not guest_id:
        guest_id = str(uuid.uuid4())
        response.set_cookie(
            key="rex_guest_session",
            value=guest_id,
            httponly=True,
            samesite="lax",
            max_age=3600 * 24 * 365 # 1 year
        )
        
    limit = int(os.getenv("GUEST_DAILY_LIMIT", "10"))
    
    if rex_user_session:
        from web.db import get_user_by_session_id
        user = get_user_by_session_id(rex_user_session)
        if user:
            return {
                "authenticated": True,
                "mode": "user",
                "user": {
                    "id": user["id"],
                    "name": user["name"],
                    "email": user["email"],
                    "avatar": user["avatar_url"]
                },
                "guest_session": guest_id
            }
            
    # Fallback to guest mode
    from web.db import get_guest_usage_today
    usage = get_guest_usage_today(guest_id)
    return {
        "authenticated": False,
        "mode": "guest",
        "user": None,
        "guest_session": guest_id,
        "usage": usage,
        "limit": limit
    }

# Server-side OAuth state store (avoids SameSite cookie issues during cross-domain redirect)
_oauth_states: dict = {}

@app.get("/api/auth/google")
def google_login(
    rex_guest_session: Optional[str] = Cookie(None)
):
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if not client_id or "your-google-client-id" in (client_id or ""):
        raise HTTPException(status_code=500, detail="Google OAuth not configured. Set GOOGLE_CLIENT_ID in .env")
    if not redirect_uri:
        raise HTTPException(status_code=500, detail="GOOGLE_REDIRECT_URI not configured in .env")

    state = secrets.token_urlsafe(32)
    # Store state server-side keyed by state token (expires implicitly when used)
    _oauth_states[state] = {
        "guest_session": rex_guest_session,
        "created_at": __import__('time').time()
    }

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account"
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"
    return RedirectResponse(url=auth_url)

@app.get("/api/auth/google/callback")
def google_callback(
    code: str,
    state: str,
    error: Optional[str] = None,
    rex_guest_session: Optional[str] = Cookie(None)
):
    # Handle OAuth errors from Google (e.g. user cancelled)
    if error:
        return RedirectResponse(url="/?auth_error=" + error)

    # Validate state server-side (immune to SameSite cookie issues)
    state_data = _oauth_states.pop(state, None)
    if not state_data:
        return HTMLResponse(
            content="<h3>Login gagal: Session expired or invalid. <a href='/'>Kembali ke REXA</a></h3>",
            status_code=400
        )

    # Use guest session from state store (captured before redirect)
    guest_session_id = state_data.get("guest_session") or rex_guest_session
        
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    
    try:
        with httpx.Client() as client:
            token_res = client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code"
                }
            )
            token_data = token_res.json()
            if "error" in token_data:
                return HTMLResponse(content=f"<h3>Google login gagal: {token_data.get('error_description', 'Token exchange failed')}</h3>", status_code=400)
            
            access_token = token_data["access_token"]
            
            user_res = client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            user_data = user_res.json()
    except Exception as e:
        return HTMLResponse(content=f"<h3>Google login gagal. Silakan coba lagi.</h3>", status_code=500)
        
    google_id = user_data.get("sub")
    name = user_data.get("name")
    email = user_data.get("email")
    avatar = user_data.get("picture")
    
    if not google_id or not email:
        return HTMLResponse(content="<h3>Google login gagal: Profile information incomplete.</h3>", status_code=400)
        
    from web.db import get_user_by_google_id, create_user, create_session, migrate_guest_chats
    user = get_user_by_google_id(google_id)
    if user:
        user_id = user["id"]
    else:
        user_id = create_user(google_id, name, email, avatar)
        
    session_id = create_session(user_id)
    
    if guest_session_id:
        migrate_guest_chats(rex_guest_session, user_id)
        
    frontend_url = os.getenv("FRONTEND_URL", "/")
    res = RedirectResponse(url=frontend_url)
    res.set_cookie(
        key="rex_user_session",
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=3600 * 24 * 7 # 7 days
    )
    res.delete_cookie("google_oauth_state")
    return res

@app.post("/api/auth/logout")
def auth_logout(response: Response, rex_user_session: Optional[str] = Cookie(None)):
    if rex_user_session:
        from web.db import delete_session
        try:
            delete_session(rex_user_session)
        except Exception:
            pass
    response.delete_cookie("rex_user_session")
    return {"status": "success"}

# --- History & Conversations Endpoints ---

@app.get("/api/conversations")
def get_conversations(
    rex_user_session: Optional[str] = Cookie(None),
    rex_guest_session: Optional[str] = Cookie(None)
):
    from web.db import get_user_by_session_id, get_user_conversations, get_guest_conversations
    if rex_user_session:
        user = get_user_by_session_id(rex_user_session)
        if user:
            return get_user_conversations(user["id"])
    if rex_guest_session:
        return get_guest_conversations(rex_guest_session)
    return []

@app.get("/api/conversations/{conv_id}")
def get_conv_messages(
    conv_id: str,
    rex_user_session: Optional[str] = Cookie(None),
    rex_guest_session: Optional[str] = Cookie(None)
):
    from web.db import get_conversation, get_conversation_messages, get_user_by_session_id
    conv = get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conv["user_id"]:
        if not rex_user_session:
            raise HTTPException(status_code=403, detail="Forbidden")
        user = get_user_by_session_id(rex_user_session)
        if not user or user["id"] != conv["user_id"]:
            raise HTTPException(status_code=403, detail="Forbidden")
    else:
        if not rex_guest_session or conv["guest_session_id"] != rex_guest_session:
            raise HTTPException(status_code=403, detail="Forbidden")
            
    return get_conversation_messages(conv_id)

@app.delete("/api/conversations/{conv_id}")
def delete_conv(
    conv_id: str,
    rex_user_session: Optional[str] = Cookie(None),
    rex_guest_session: Optional[str] = Cookie(None)
):
    from web.db import get_conversation, delete_conversation, get_user_by_session_id
    conv = get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    if conv["user_id"]:
        if not rex_user_session:
            raise HTTPException(status_code=403, detail="Forbidden")
        user = get_user_by_session_id(rex_user_session)
        if not user or user["id"] != conv["user_id"]:
            raise HTTPException(status_code=403, detail="Forbidden")
    else:
        if not rex_guest_session or conv["guest_session_id"] != rex_guest_session:
            raise HTTPException(status_code=403, detail="Forbidden")
            
    delete_conversation(conv_id)
    return {"status": "success"}

# --- Chat Endpoint ---

@app.get("/api/health")
def health():
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_configured = gemini_api_key is not None and gemini_api_key.strip() != ""
    return {
        "status": "ok",
        "model": active_model_version,
        "device": "cpu",
        "local_model": local_model_loaded,
        "gemini_configured": gemini_configured
    }

@app.post("/api/chat")
def chat(
    request: ChatRequest,
    response: Response,
    rex_user_session: Optional[str] = Cookie(None),
    rex_guest_session: Optional[str] = Cookie(None)
):
    selected_model = request.model or "local"
    
    # 1. Resolve User or Guest identity
    user_id = None
    guest_id = rex_guest_session
    
    if rex_user_session:
        from web.db import get_user_by_session_id
        user = get_user_by_session_id(rex_user_session)
        if user:
            user_id = user["id"]
            
    if not user_id:
        # Enforce Guest limit check
        if not guest_id:
            guest_id = str(uuid.uuid4())
            response.set_cookie(
                key="rex_guest_session",
                value=guest_id,
                httponly=True,
                samesite="lax",
                max_age=3600 * 24 * 365
            )
            
        from web.db import get_guest_usage_today, increment_guest_usage
        usage = get_guest_usage_today(guest_id)
        limit = int(os.getenv("GUEST_DAILY_LIMIT", "10"))
        
        if usage >= limit:
            return JSONResponse(
                status_code=403,
                content={"error": "Guest limit kamu sudah habis. Login dengan Google untuk melanjutkan."}
            )
        
        increment_guest_usage(guest_id)
        
    # 2. Extract input messages
    current_message = request.message
    chat_history = request.history or []
    
    if not current_message and request.messages:
        current_message = request.messages[-1].content
        chat_history = request.messages[:-1]
    elif not chat_history and request.messages:
        if request.messages[-1].content == current_message:
            chat_history = request.messages[:-1]
        else:
            chat_history = request.messages
            
    if not current_message or current_message.strip() == "":
        return {"response": "", "model": selected_model}

    # 3. Resolve or Create Conversation in DB
    from web.db import get_conversation, create_conversation, create_message
    conv_id = request.conversation_id
    
    if conv_id:
        conv = get_conversation(conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        # Validate ownership
        if conv["user_id"] and conv["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Forbidden")
        if not conv["user_id"] and conv["guest_session_id"] != guest_id:
            raise HTTPException(status_code=403, detail="Forbidden")
    else:
        title = current_message.strip()
        if len(title) > 30:
            title = title[:30] + "..."
        conv_id = create_conversation(user_id=user_id, guest_session_id=guest_id, title=title)
        
    # 4. Save user message to database
    create_message(conv_id, role="user", content=current_message)

    # 5. Model Inference
    response_text = ""
    if selected_model == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key or gemini_api_key.strip() == "":
            return JSONResponse(status_code=400, content={"error": "Gemini API key is not configured."})
            
        gemini_model = os.getenv("GEMINI_MODEL") or "gemini-flash-latest"
        
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=gemini_api_key)
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=request.temperature if request.temperature is not None else 0.7,
            )
            
            contents = []
            for msg in chat_history:
                role = 'model' if msg.role == 'assistant' else 'user'
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.content or "")]
                ))
                
            contents.append(types.Content(
                role='user',
                parts=[types.Part.from_text(text=current_message)]
            ))
            
            res_gemini = client.models.generate_content(
                model=gemini_model,
                contents=contents,
                config=config
            )
            response_text = res_gemini.text or ""
        except Exception as e:
            err_str = str(e)
            print(f"Gemini API connection error: {err_str}")
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "quota" in err_str.lower():
                return JSONResponse(
                    status_code=429,
                    content={"error": "Quota Gemini API sedang habis/limit. Silakan ganti ke model REXA 1.0 (Local) di dropdown input."}
                )
            return JSONResponse(status_code=500, content={"error": f"Gemini error: {err_str[:120]}"})
            
    else:
        # Use Local model
        if not local_model_loaded or generator is None:
            raise HTTPException(status_code=503, detail="Local model is not loaded or failed to initialize.")
            
        if active_model_version == "V5":
            prompt = "<|system|>Kamu adalah REXA, asisten AI dari REXSHIN."
            for msg in chat_history:
                if msg.role == "user":
                    prompt += f"<|user|>{msg.content}"
                elif msg.role == "assistant":
                    prompt += f"<|assistant|>{msg.content}"
            prompt += f"<|user|>{current_message}<|assistant|>"
        else:
            if len(chat_history) > 0:
                prompt = " ".join([m.content for m in chat_history]) + " " + current_message
            else:
                prompt = current_message
                
        try:
            temp = request.temperature if request.temperature is not None else 0.7
            k = request.top_k if request.top_k is not None else 10
            tokens = request.max_new_tokens if request.max_new_tokens is not None else 50
            
            response_text, _ = generator.generate(
                prompt=prompt,
                max_new_tokens=tokens,
                temperature=temp,
                top_k=k,
                add_special_tokens=True
            )
            
            if active_model_version == "V5":
                for tag in ["<|system|>", "<|user|>", "<|assistant|>", "<|eos|>", "<eos>"]:
                    response_text = response_text.replace(tag, "")
                    response_text = response_text.replace(tag.upper(), "")
                response_text = response_text.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Local generation failed: {str(e)}")
            
    # 6. Save model response to database
    create_message(conv_id, role="assistant", content=response_text, model=selected_model)
    
    return {
        "response": response_text,
        "model": selected_model,
        "conversation_id": conv_id
    }

# Mount static files directory
app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/")
def get_root():
    return FileResponse("web/static/index.html")
