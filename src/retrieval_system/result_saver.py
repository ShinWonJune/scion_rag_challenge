# -*- coding: utf-8 -*-
"""Handles saving retrieval results to files."""

from __future__ import annotations

import json
import os
import uuid

from retrieval_system import utils


class ResultSaver:
    """Manages the creation of output directories and saving of JSON results."""

    def __init__(self, output_root: str, output_subdir: str | None = None):
        self.output_root = output_root
        if output_subdir is None:
            folder = utils.timestamp_folder_kst()
        else:
            folder = output_subdir
        self.output_dir = os.path.join(self.output_root, folder) if folder else self.output_root
        utils.ensure_dir(self.output_dir)

    def save_result(self, payload: dict[str, any]) -> str:
        """
        Saves a single retrieval result payload to a JSON file.

        The filename is constructed from the question ID, model name, and a unique trial ID.

        Args:
            payload (dict): The dictionary containing the result data.
                           Must include 'id', and 'model_name'.

        Returns:
            str: The full path to the saved file.
        """
        qid = payload.get("id")
        model_name = payload.get("model_name")
        if not qid or not model_name:
            raise ValueError("Payload must contain 'id' and 'model_name' keys.")

        trial_id = uuid.uuid4().hex[:8]
        safe_model_name = utils.safe_model_name_for_path(model_name)

        # Add metadata to the payload before saving
        payload["timestamp_kst"] = utils.now_kst().isoformat()
        payload["trial_id"] = trial_id

        filename = f"{qid}_{safe_model_name}_{trial_id}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return filepath

    def get_output_folder(self) -> str:
        """Returns the full path to the output directory for this run."""
        return self.output_dir
