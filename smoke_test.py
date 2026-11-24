"""Smoke test: launch the Gradio app headlessly and assert the HTTP endpoint is up.

Run from the project root:  python smoke_test.py
"""

import requests

from app import demo

PORT = 7861


def main() -> None:
    demo.launch(server_name="127.0.0.1", server_port=PORT,
                prevent_thread_lock=True, quiet=True)
    try:
        resp = requests.get(f"http://127.0.0.1:{PORT}/", timeout=10)
        assert resp.status_code == 200, f"unexpected status {resp.status_code}"
        assert "gradio" in resp.text.lower() or "RegIntel" in resp.text
        print(f"SMOKE OK: GET http://127.0.0.1:{PORT}/ -> {resp.status_code}")
    finally:
        demo.close()


if __name__ == "__main__":
    main()
