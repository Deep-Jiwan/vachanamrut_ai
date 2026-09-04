"""Dependency-free diagnostic endpoint.

Uses nothing but the standard library, so it answers the one question a failing
function cannot answer about itself: is the Python runtime working at all, were
the third-party dependencies installed, and did the corpus files make it into
the bundle. If this responds while /api/health does not, the fault is in the
dependencies or the bundle rather than in the runtime.

Safe to delete once the deployment is healthy.
"""
from http.server import BaseHTTPRequestHandler
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def probe_import(name: str) -> str:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "ok")
    except Exception as exc:
        return f"FAILED: {type(exc).__name__}: {exc}"


def listing(path: Path, limit: int = 40) -> list[str]:
    try:
        return sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())[:limit]
    except Exception as exc:
        return [f"FAILED: {exc}"]


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        sys.path.insert(0, str(ROOT / "src"))
        report = {
            "python": sys.version,
            "cwd": os.getcwd(),
            "file": __file__,
            "root": str(ROOT),
            "root_entries": listing(ROOT),
            "src_exists": (ROOT / "src").is_dir(),
            "src_entries": listing(ROOT / "src"),
            "corpus_exists": (ROOT / "data" / "corpus").is_dir(),
            "corpus_entries": listing(ROOT / "data" / "corpus"),
            "imports": {
                "fastapi": probe_import("fastapi"),
                "httpx": probe_import("httpx"),
                "pydantic": probe_import("pydantic"),
                "vachanamrut_rag": probe_import("vachanamrut_rag"),
            },
        }
        body = json.dumps(report, indent=1).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
