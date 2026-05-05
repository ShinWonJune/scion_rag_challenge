"""Codex CLI client for SHRAG generation pipeline.

Drop-in replacement matching OpenAIClient.generate_answer_with_prompt(prompt, max_tokens).
Codex CLI uses ChatGPT subscription auth, bypassing OpenAI API quota.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from typing import Any


class CodexClient:
    def __init__(
        self,
        model: str = "gpt-5.4",
        cwd: str | None = None,
        sandbox: str = "read-only",
        timeout_sec: int = 600,
    ) -> None:
        self.model = model
        self.cwd = cwd
        self.sandbox = sandbox
        self.timeout_sec = timeout_sec

        codex_path = shutil.which("codex")
        if not codex_path:
            raise RuntimeError("`codex` CLI not found on PATH. Install via `npm install -g @openai/codex`.")
        self.codex_path = codex_path

    def generate_answer_with_prompt(self, prompt: str, max_tokens: int = 4000) -> str:
        # max_tokens unused by codex CLI (it uses model-internal limits) but kept for API parity.
        del max_tokens

        with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False, encoding="utf-8") as f:
            out_path = f.name

        cmd: list[str] = [
            self.codex_path,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--dangerously-bypass-approvals-and-sandbox",
            "-s",
            self.sandbox,
            "-m",
            self.model,
            "--output-last-message",
            out_path,
        ]

        attempts = 0
        max_attempts = 3
        last_error: Exception | None = None
        while attempts < max_attempts:
            attempts += 1
            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_sec,
                    cwd=self.cwd,
                )
                if proc.returncode != 0:
                    snippet = (proc.stderr or proc.stdout or "")[-500:]
                    raise RuntimeError(f"codex exec returncode={proc.returncode}: {snippet}")
                with open(out_path, "r", encoding="utf-8") as f:
                    answer = f.read().strip()
                if not answer:
                    snippet = (proc.stdout or "")[-500:]
                    raise RuntimeError(f"codex exec returned empty answer; tail: {snippet}")
                return answer
            except subprocess.TimeoutExpired as exc:
                last_error = exc
                if attempts >= max_attempts:
                    break
                time.sleep(5.0 * attempts)
            except RuntimeError as exc:
                last_error = exc
                if attempts >= max_attempts:
                    break
                time.sleep(5.0 * attempts)
            finally:
                pass

        try:
            os.unlink(out_path)
        except OSError:
            pass

        assert last_error is not None
        raise last_error
