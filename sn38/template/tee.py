"""
TEE authentication via RA-TLS.

The validator obtains a TLS certificate with attestation embedded (RA-TLS)
from the dstack SDK. This cert is used as a client certificate for mTLS
with the backend. The attestation is in X.509 extensions — no separate
headers needed, TLS handles replay protection natively.

Retries indefinitely on backend downtime.
"""

import os
import tempfile

import bittensor as bt
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ValidatorSession:

    def __init__(self, backend_url: str, hotkey: str = "unknown"):
        self.backend_url = backend_url
        self._cert_path = None
        self._key_path = None

        try:
            from dstack_sdk import DstackClient
            client = DstackClient(timeout=120)
            result = client.get_tls_key(
                subject=hotkey,
                usage_ra_tls=True,
                usage_client_auth=True,
                with_app_info=True,
            )

            cert_pem = "\n".join(result.certificate_chain)
            key_pem = result.key

            cert_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False, mode="w")
            cert_file.write(cert_pem)
            cert_file.close()
            self._cert_path = cert_file.name

            key_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False, mode="w")
            key_file.write(key_pem)
            key_file.close()
            self._key_path = key_file.name
        except Exception as e:
            bt.logging.error(f"[TEE] get_tls_key failed: {type(e).__name__}: {e}")

        self.session = requests.Session()
        if self._cert_path and self._key_path:
            self.session.cert = (self._cert_path, self._key_path)
        retry = Retry(
            total=None,
            backoff_factor=5,
            backoff_max=300,
            status_forcelist=[502, 503, 504],
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    @property
    def is_tee(self) -> bool:
        return self._cert_path is not None

    def _check_forbidden(self, resp: requests.Response):
        if resp.status_code == 403:
            detail = resp.json().get("detail", "Unknown reason")
            bt.logging.error(f"Backend rejected this validator (403): {detail}")
            bt.logging.error("Your TEE image is not authorized. Update your validator or contact the subnet owner.")
            raise SystemExit(1)

    def get(self, path: str) -> requests.Response:
        resp = self.session.get(f"{self.backend_url}{path}")
        self._check_forbidden(resp)
        return resp

    def post(self, path: str, json_data=None) -> requests.Response:
        resp = self.session.post(f"{self.backend_url}{path}", json=json_data)
        self._check_forbidden(resp)
        return resp
