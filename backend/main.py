from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.data_loader import load_dataframes
<<<<<<< HEAD
from backend.routers import search, vyroky, clanky, stats, video
=======
from backend.routers import search, vyroky, clanky, stats, politicians, parties, oblasts, verify, statements, chat, dashboard
>>>>>>> bec38623368a43fe3e27bd5d69c1cc6171401ea2


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dataframes()
    yield


app = FastAPI(
    title="Demagog API",
    description="REST API for Slovak political fact-checking data (demagog.sk)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)
app.include_router(vyroky.router)
app.include_router(clanky.router)
app.include_router(stats.router)
<<<<<<< HEAD
app.include_router(video.router)
=======
app.include_router(politicians.router)
app.include_router(parties.router)
app.include_router(oblasts.router)
app.include_router(verify.router)
app.include_router(statements.router)
app.include_router(chat.router)
app.include_router(dashboard.router)
>>>>>>> bec38623368a43fe3e27bd5d69c1cc6171401ea2


@app.get("/")
def root():
    return {"status": "ok", "service": "Demagog API"}

