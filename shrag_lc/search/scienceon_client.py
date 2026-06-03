"""Low-level ScienceON API client (AES-CBC token management + article search).

Clean port of ``shrag/search/clients/scienceon_api_example.py`` — drops the
module-level atexit/elapsed-time side effects, keeps the credential/token
lifecycle and XML response parsing intact. Requires a credentials JSON with
client_id / auth_key / mac_address / refresh tokens.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

logger = logging.getLogger(__name__)

BASE_URL = "https://apigateway.kisti.re.kr/openapicall.do"
TOKEN_REQUEST_URL = "https://apigateway.kisti.re.kr/tokenrequest.do"
TOKEN_EXPIRY_BUFFER = timedelta(minutes=1)
REQUEST_TIMEOUT_SEC = 30


class AESCipher:
    def __init__(self, auth_key: str):
        if len(auth_key) != 32:
            raise ValueError("API key must be 32 bytes.")
        self.key = auth_key.encode("utf-8")
        self.block_size = AES.block_size
        self.iv = "jvHJ1EFA0IXBrxxz".encode("utf-8")

    def encrypt(self, plain_text: str) -> str:
        cipher = AES.new(self.key, AES.MODE_CBC, self.iv)
        encrypted = cipher.encrypt(pad(plain_text.encode("utf-8"), self.block_size))
        return base64.urlsafe_b64encode(encrypted).decode("utf-8")


class CredentialManager:
    def __init__(self, credentials_path: Path):
        self.credentials_path = credentials_path
        with open(credentials_path, "r", encoding="utf-8") as f:
            self.credentials = json.load(f)
        self.aes_cipher = AESCipher(self.auth_key)
        self.lock = Lock()

    def _save(self) -> None:
        with open(self.credentials_path, "w", encoding="utf-8") as f:
            json.dump(self.credentials, f, indent=4)

    @property
    def mac_address(self) -> str:
        return self.credentials.get("mac_address")

    @property
    def auth_key(self) -> str:
        return self.credentials.get("auth_key")

    @property
    def client_id(self) -> str:
        return self.credentials.get("client_id")

    @property
    def access_token(self) -> str:
        return self.credentials.get("access_token")

    @property
    def refresh_token(self) -> str:
        return self.credentials.get("refresh_token")

    def _is_token_valid(self, expiry_str: str | None) -> bool:
        if not expiry_str:
            return False
        try:
            raw = expiry_str.split(".")[0] if "." in expiry_str else expiry_str
            expire_time = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
            return datetime.now() < (expire_time - TOKEN_EXPIRY_BUFFER)
        except (ValueError, TypeError):
            logger.warning("Invalid expiration time format: %s", expiry_str)
            return False

    def _update_tokens(self, token_data: dict) -> None:
        self.credentials.update(token_data)
        self._save()

    def _request_new_tokens(self, session: requests.Session) -> None:
        now_str = "".join(re.findall(r"\d", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        payload = json.dumps({"datetime": now_str, "mac_address": self.mac_address}).replace(" ", "")
        params = {"client_id": self.client_id, "accounts": self.aes_cipher.encrypt(payload)}
        with session.get(TOKEN_REQUEST_URL, params=params, timeout=REQUEST_TIMEOUT_SEC) as resp:
            resp.raise_for_status()
            self._update_tokens(resp.json())

    def get_access_token(self, session: requests.Session) -> str:
        with self.lock:
            if self._is_token_valid(self.credentials.get("access_token_expire")):
                return self.access_token
            logger.warning("Access token expired; requesting new tokens.")
            self._request_new_tokens(session)
            return self.access_token


class ScienceONAPIClient:
    def __init__(self, credentials_path: Path):
        self.credential_manager = CredentialManager(credentials_path)
        self.session = requests.Session()

    def close_session(self) -> None:
        self.session.close()

    @staticmethod
    def _parse_search_response(xml_text: str, fields: list[str]) -> list[dict]:
        root = ET.fromstring(xml_text)
        records: list[dict] = []
        field_map = {
            "CN": {"metaCode": "CN"},
            "title": {"metaName": "논문명"},
            "abstract": {"metaName": "초록"},
            "author": {"metaName": "저자"},
            "link": {"metaName": "ScienceON상세링크"},
            "publisher": {"metaName": "출판사(발행기관)"},
            "journal": {"metaName": "저널명"},
            "year": {"metaName": "발행년"},
        }
        requested = {k: v for k, v in field_map.items() if k in fields}
        record_list = root.find("recordList")
        if record_list is None:
            return records
        for record in record_list.findall("record"):
            record_dict: dict = {}
            for item in record.findall("item"):
                text = item.text.strip() if item.text else ""
                for field_name, criteria in requested.items():
                    if all(item.attrib.get(k) == v for k, v in criteria.items()):
                        record_dict[field_name] = text
                        break
            if record_dict:
                records.append(record_dict)
        return records

    def search_articles(
        self, query: str, cur_page: int = 1, row_count: int = 10, fields: list[str] | None = None
    ) -> list[dict]:
        if fields is None:
            fields = ["title", "author", "abstract", "CN"]
        access_token = self.credential_manager.get_access_token(self.session)
        params = {
            "client_id": self.credential_manager.client_id,
            "token": access_token,
            "version": "1.0",
            "action": "search",
            "target": "ARTI",
            "searchQuery": f'{{"BI":"{query}"}}',
            "sortField": "",
            "curPage": str(cur_page),
            "rowCount": str(row_count),
            "session_id": "",
            "include": "",
            "grouping": "",
        }
        try:
            with self.session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SEC) as resp:
                resp.raise_for_status()
                return self._parse_search_response(resp.text, fields)
        except requests.RequestException as e:
            logger.error("ScienceON API request failed: %s", e)
            return []
