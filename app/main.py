from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from morph.engine import MorphEngine
from .routes import router

WEB_DIR = Path("web")
RENDER_DIR = Path(".cache/render")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = MorphEngine()
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Morph", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve rendered audio files directly
app.mount("/audio", StaticFiles(directory=RENDER_DIR), name="audio")

# Optional: serve other static frontend assets like app.js / styles.css
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serve the main HTML UI.
    """
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            """
            <!doctype html>
            <html>
              <head><title>Morph</title></head>
              <body>
                <h1>Morph</h1>
                <p>UI not found. Put web/index.html in the repo root.</p>
              </body>
            </html>
            """
        )

    return HTMLResponse(index_path.read_text(encoding="utf-8"))
