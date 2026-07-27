"""A-share index/sector volume and fund-flow bar charts (Flask :5002)."""

from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data_service import get_overview


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/overview")
    def api_overview():
        force = request.args.get("force", "").lower() in ("1", "true", "yes")
        return jsonify(get_overview(force=force))

    return app


def main() -> None:
    app = create_app()
    app.run(host="127.0.0.1", port=5002, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
