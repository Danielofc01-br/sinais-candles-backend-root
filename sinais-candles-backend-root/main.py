from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils import get_latest_signals

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/sinal")
def sinal():
    sinais = get_latest_signals()
    return sinais
