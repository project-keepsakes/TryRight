import asyncio
import importlib.util
import http.client
import json
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "app" / "voice"))
from provider_safety import PaidServicesDisabled, require_paid_services


class ProviderSafetyTests(unittest.TestCase):
    def test_credentials_alone_do_not_enable_spending(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "synthetic-key"}, clear=True):
            with self.assertRaises(PaidServicesDisabled):
                require_paid_services()

    def test_local_opt_in_is_exact_and_never_enables_hosted_spending(self):
        for value in ("false", "1", "TRUE", ""):
            with self.subTest(value=value), patch.dict(os.environ, {"TRIRIGHT_ALLOW_PAID_SERVICES": value}, clear=True):
                with self.assertRaises(PaidServicesDisabled):
                    require_paid_services()
        with patch.dict(os.environ, {"TRIRIGHT_ALLOW_PAID_SERVICES": "true"}, clear=True):
            require_paid_services()
        with patch.dict(os.environ, {"TRIRIGHT_ALLOW_PAID_SERVICES": "true", "VERCEL": "1"}, clear=True):
            with self.assertRaises(PaidServicesDisabled):
                require_paid_services()

    @classmethod
    def setUpClass(cls):
        original_exists = Path.exists
        def exists(path):
            return False if path.name in (".env", ".cache.json", "voice-cache.json") else original_exists(path)
        with patch.dict(os.environ, {"VERCEL": "1"}, clear=True), patch.object(Path, "exists", exists), patch("dotenv.load_dotenv"):
            spec = importlib.util.spec_from_file_location("trial_app_test", ROOT / "app" / "app.py")
            cls.trial = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.trial)
            import server
            cls.voice = server
            spec = importlib.util.spec_from_file_location("voice_entry_test", ROOT / "app" / "api" / "voice.py")
            entry = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(entry)
            cls.mounted_voice = entry.app

    def test_trial_http_rejects_paid_requests_before_parsing_or_provider_calls(self):
        server = self.trial.ThreadingHTTPServer(("127.0.0.1", 0), self.trial.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {"VERCEL": "1", "TRIRIGHT_ALLOW_PAID_SERVICES": "true"}, clear=True), patch.object(self.trial, "get_client") as client:
                for route in ("match", "screen", "summarize"):
                    request = Request(f"http://127.0.0.1:{server.server_port}/api/{route}", data=b"not json", method="POST")
                    with self.assertRaises(HTTPError) as response:
                        urlopen(request, timeout=3)
                    self.assertEqual(response.exception.code, 503)
                client.assert_not_called()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_direct_provider_wrappers_reject_calls_even_with_cached_clients(self):
        import claude_text
        import eleven
        with patch.dict(os.environ, {}, clear=True), patch.object(claude_text, "_client", object()), patch.object(eleven._http, "request") as request:
            for call in (self.trial.get_client, claude_text.client, eleven._key, lambda: eleven.download("https://example.invalid/audio")):
                with self.assertRaises(RuntimeError):
                    call()
            request.assert_not_called()

    def test_trial_http_rejects_invalid_framing_without_reading_a_body(self):
        server = self.trial.ThreadingHTTPServer(("127.0.0.1", 0), self.trial.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {"TRIRIGHT_ALLOW_PAID_SERVICES": "true"}, clear=True), patch.object(self.trial, "get_client") as provider:
                for length, expected in [("9" * 4500, 400), ("²", 400), ("-1", 400), ("16385", 413)]:
                    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                    try:
                        connection.putrequest("POST", "/api/match")
                        connection.putheader("Content-Length", length)
                        connection.endheaders()
                        self.assertEqual(connection.getresponse().status, expected)
                    finally:
                        connection.close()
                provider.assert_not_called()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_voice_routes_reject_before_body_reads_and_provider_calls(self):
        import httpx
        async def check():
            for app, prefix in [(self.voice.app, ""), (self.mounted_voice, "/voice")]:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    for method, path in [("POST", "/api/trials/example/script"), ("POST", "/api/dub"), ("GET", "/api/agents/qa/session"), ("POST", "/api/teachback/grade")]:
                        response = await client.request(method, prefix + path, content=b"not json")
                        self.assertEqual(response.status_code, 503)
                        self.assertEqual(response.json(), {"error": "Paid services are disabled"})
                    self.assertEqual((await client.get(prefix + "/api/config")).status_code, 200)
        with patch.dict(os.environ, {}, clear=True), patch.object(self.voice.claude_text, "client") as client, patch.object(self.voice.eleven._http, "request") as request:
            asyncio.run(check())
            client.assert_not_called()
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
