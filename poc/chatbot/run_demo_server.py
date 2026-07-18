"""Demo-video launcher: chatbot on :8003 with MIRA_STRIP_EMOJI=1 (the
simulator runtime's emoji font is broken; see miranote-demo/PLAN.md)."""

import os
import sys
from pathlib import Path

os.environ["MIRA_STRIP_EMOJI"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import uvicorn

uvicorn.run("poc.chatbot.main:app", port=8003, log_level="warning")
