from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from gorgias import service


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(service.api)

    def test_health_does_not_require_ollama(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_extraction_contract_reports_non_completeness(self):
        with patch.object(service, "_model", return_value=object()), \
             patch.object(service.extraction, "annotate", return_value="rule(x)."):
            response = self.client.post(
                "/v1/extractions",
                json={"text": "Si A, alors B.", "format": "lpp", "language": "fr"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["result"], "rule(x).")
        self.assertFalse(body["completeness_guaranteed"])
        self.assertEqual(body["language"], "fr")
        self.assertEqual(len(body["source_sha256"]), 64)

    def test_unvalidated_language_is_refused(self):
        response = self.client.post(
            "/v1/extractions", json={"text": "If A then B", "language": "en"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
