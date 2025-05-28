import json
import os
import time
import uuid
import threading
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Configuration
TYPETHINK_API_URL = "https://chat.typethink.ai/api/chat/completions"

# Global variables for client API keys and Typethink tokens
VALID_CLIENT_KEYS: set = set()
TYPETHINK_TOKENS: list = []
current_typethink_token_index: int = 0
token_rotation_lock = threading.Lock()

# Pydantic Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

class StreamChoice(BaseModel):
    delta: Dict[str, Any] = Field(default_factory=dict)
    index: int = 0
    finish_reason: Optional[str] = None

class StreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]

# FastAPI App
app = FastAPI(title="Typethink OpenAI API")
security = HTTPBearer(auto_error=False)

# Global variables
models_data = {}

def load_models():
    """Load models from models.json"""
    try:
        with open("models.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading models.json: {e}")
        return {"data": []}

def load_client_api_keys():
    """Load client API keys from client_api_keys.json"""
    global VALID_CLIENT_KEYS
    try:
        with open("client_api_keys.json", "r", encoding="utf-8") as f:
            keys = json.load(f)
            if not isinstance(keys, list):
                print("Warning: client_api_keys.json should contain a list of keys.")
                VALID_CLIENT_KEYS = set()
                return
            VALID_CLIENT_KEYS = set(keys)
            if not VALID_CLIENT_KEYS:
                print("Warning: client_api_keys.json is empty.")
            else:
                print(f"Successfully loaded {len(VALID_CLIENT_KEYS)} client API keys.")
    except FileNotFoundError:
        print("Error: client_api_keys.json not found.")
        VALID_CLIENT_KEYS = set()
    except Exception as e:
        print(f"Error loading client_api_keys.json: {e}")
        VALID_CLIENT_KEYS = set()

def load_typethink_tokens():
    """Load Typethink tokens from accounts.json"""
    global TYPETHINK_TOKENS
    try:
        with open("accounts.json", "r", encoding="utf-8") as f:
            accounts_data = json.load(f)
            if not isinstance(accounts_data, list):
                print("Warning: accounts.json should contain a list of accounts.")
                TYPETHINK_TOKENS = []
                return

            loaded_tokens = []
            for account in accounts_data:
                token = account.get("websocket_cookies", {}).get("token")
                if token and isinstance(token, str):
                    loaded_tokens.append(token)
            
            TYPETHINK_TOKENS = loaded_tokens
            if not TYPETHINK_TOKENS:
                print("Warning: No valid tokens found in accounts.json.")
            else:
                print(f"Successfully loaded {len(TYPETHINK_TOKENS)} Typethink tokens.")

    except FileNotFoundError:
        print("Error: accounts.json not found.")
        TYPETHINK_TOKENS = []
    except Exception as e:
        print(f"Error loading accounts.json: {e}")
        TYPETHINK_TOKENS = []

def get_model_item(model_id: str) -> Optional[Dict]:
    """Get model item by ID from loaded models data"""
    for model in models_data.get("data", []):
        if model.get("id") == model_id:
            return model
    return None

async def authenticate_client(auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Authenticate client based on API key in Authorization header"""
    if not VALID_CLIENT_KEYS:
        raise HTTPException(status_code=503, detail="Service unavailable: No client API keys configured.")
    
    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=401,
            detail="API key required in Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if auth.credentials not in VALID_CLIENT_KEYS:
        raise HTTPException(status_code=403, detail="Invalid client API key.")

def get_next_typethink_token() -> str:
    """Get the next Typethink token using round-robin"""
    global current_typethink_token_index
    
    if not TYPETHINK_TOKENS:
        raise HTTPException(status_code=503, detail="Service unavailable: No Typethink tokens configured.")
    
    with token_rotation_lock:
        if not TYPETHINK_TOKENS:
            raise HTTPException(status_code=503, detail="Service unavailable: Typethink tokens unavailable.")
        token_to_use = TYPETHINK_TOKENS[current_typethink_token_index]
        current_typethink_token_index = (current_typethink_token_index + 1) % len(TYPETHINK_TOKENS)
    return token_to_use

@app.on_event("startup")
async def startup():
    global models_data
    models_data = load_models()
    load_client_api_keys()
    load_typethink_tokens()

@app.get("/v1/models", response_model=ModelList)
async def list_models(_: None = Depends(authenticate_client)):
    """List available models"""
    model_list = []
    for model in models_data.get("data", []):
        model_list.append(ModelInfo(
            id=model.get("id", ""),
            created=model.get("created", int(time.time())),
            owned_by=model.get("owned_by", "typethink")
        ))
    return ModelList(data=model_list)

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _: None = Depends(authenticate_client)
):
    """Create chat completion"""
    model_item = get_model_item(request.model)
    if not model_item:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
    
    typethink_token = get_next_typethink_token()
    
    payload = {
        "stream": True,
        "model": request.model,
        "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
        "params": {},
        "tool_servers": [],
        "features": {
            "image_generation": False,
            "code_interpreter": False,
            "web_search": False,
        },
        "model_item": model_item,
        "background_tasks": {"title_generation": False, "tags_generation": False},
    }
    
    if request.temperature is not None:
        payload["params"]["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["params"]["max_tokens"] = request.max_tokens
    if request.top_p is not None:
        payload["params"]["top_p"] = request.top_p
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Content-Type": "application/json",
        "sec-ch-ua-platform": '"Windows"',
        "authorization": f"Bearer {typethink_token}",
        "origin": "https://chat.typethink.ai",
    }
    
    if request.stream:
        return StreamingResponse(
            stream_generator(payload, headers, request.model),
            media_type="text/event-stream"
        )
    else:
        return await non_stream_response(payload, headers, request.model)

async def stream_generator(payload: Dict, headers: Dict, model: str) -> AsyncGenerator[str, None]:
    """Generate streaming response"""
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", TYPETHINK_API_URL, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    error_msg = await response.aread()
                    yield f'data: {{"error": "API Error: {error_msg.decode()}"}}\n\n'
                    yield "data: [DONE]\n\n"
                    return
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        
                        try:
                            event = json.loads(data)
                            # Convert to OpenAI format
                            delta = {}
                            finish_reason = None
                            
                            if event.get("choices") and len(event["choices"]) > 0:
                                choice = event["choices"][0]
                                if choice.get("delta"):
                                    delta = choice["delta"]
                                if choice.get("finish_reason"):
                                    finish_reason = choice["finish_reason"]
                            
                            stream_resp = StreamResponse(
                                model=model,
                                choices=[StreamChoice(delta=delta, finish_reason=finish_reason)]
                            )
                            yield f"data: {stream_resp.json()}\n\n"
                            
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            print(f"Stream error: {e}")
                            continue
    except Exception as e:
        yield f'data: {{"error": "Stream error: {str(e)}"}}\n\n'
        yield "data: [DONE]\n\n"

async def non_stream_response(payload: Dict, headers: Dict, model: str) -> ChatCompletionResponse:
    """Generate non-streaming response"""
    content_parts = []
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", TYPETHINK_API_URL, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    error_msg = await response.aread()
                    raise HTTPException(status_code=response.status_code, detail=f"API Error: {error_msg.decode()}")
                
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    
                    try:
                        event = json.loads(data)
                        if event.get("choices") and len(event["choices"]) > 0:
                            choice = event["choices"][0]
                            if choice.get("delta") and choice["delta"].get("content"):
                                content_parts.append(choice["delta"]["content"])
                    except json.JSONDecodeError:
                        continue
        
        full_content = "".join(content_parts)
        return ChatCompletionResponse(
            model=model,
            choices=[ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=full_content)
            )]
        )
    
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Upstream error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    
    dummy_models = {
        "data": [
            {
                "id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "anthropic",
                "name": "Claude Sonnet 4",
                "openai": {
                    "id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "openai",
                    "connection_type": "external"
                },
                "info": {
                    "id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                    "name": "Claude Sonnet 4",
                    "meta": {"capabilities": {"vision": True}},
                    "is_active": True
                }
            },
            {
                "id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "anthropic",
                "name": "Claude Sonnet 3.7",
                "openai": {
                    "id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "openai",
                    "connection_type": "external"
                },
                "info": {
                    "id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                    "name": "Claude Sonnet 3.7",
                    "meta": {"capabilities": {"vision": True}},
                    "is_active": True
                }
            },
            {
                "id": "azure/o3-mini",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "anthropic",
                "name": "o3-mini",
                "openai": {
                    "id": "azure/o3-mini",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "openai",
                    "connection_type": "external"
                },
                "info": {
                    "id": "azure/o3-mini",
                    "name": "o3-mini",
                    "meta": {"capabilities": {"vision": True}},
                    "is_active": True
                }
            },
            {
                "id": "azure/gpt-4.1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "anthropic",
                "name": "gpt-4.1",
                "openai": {
                    "id": "azure/gpt-4.1",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "openai",
                    "connection_type": "external"
                },
                "info": {
                    "id": "azure/gpt-4.1",
                    "name": "gpt-4.1",
                    "meta": {"capabilities": {"vision": True}},
                    "is_active": True
                }
            }
        ]
    }
    with open("models.json", "w", encoding="utf-8") as f:
        json.dump(dummy_models, f, indent=2)

    
    print("Starting Typethink OpenAI API server...")
    print("Endpoints:")
    print("  GET  /v1/models")
    print("  POST /v1/chat/completions")
    print("\nUse client API keys (sk-xxx) in Authorization header")

    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8100)
    except Exception as e:
        print(f"Error starting server: {e}")
        # 使用命令行方式
        import os
        os.system("python -m uvicorn test_typeai:app --host 0.0.0.0 --port 8100")
