# -*- coding: utf-8 -*-
"""Handles saving retrieval results to disk."""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict
from . import utils


class ResultSaver:
    """Saves retrieval results to a structured directory."""

    def __init__(self, output_root: str):
        self.output_root = output_root
        self.timestamp_folder = utils.timestamp_folder_kst()
        self.session_folder = os.path.join(self.output_root, self.timestamp_folder)
        utils.ensure_dir(self.session_folder)
        print(f"[INFO] Results will be saved in: {self.session_folder}")

    def save(
        self,
        qid: str,
        model_name: str,
        results: Dict[str, Any],
        meta: Dict[str, Any]
    ) -> str:
        """
        Saves a single question's retrieval result to a JSON file.
        Returns the path to the saved file.
        """
        trial_id = uuid.uuid4().hex[:8]
        payload = {
            "id": qid,
            "model_name": model_name,
            "timestamp_kst": utils.now_kst().isoformat(),
            "trial_id": trial_id,
            "queries": results,
            "meta": meta,
        }

        safe_model = utils.safe_model_name_for_path(model_name)
        fname = f"{qid}_{safe_model}_{trial_id}.json"
        path = os.path.join(self.session_folder, fname)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return path


