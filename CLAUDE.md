# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SHRAG (Search like Human with RAG) is a 5-step pipeline that combines a platform search (ScienceON / PubMed / Wikipedia) with dense-retrieval RAG:
question → keyword extraction + platform search → embed/build vectorDB → dense retrieve → LLM answer.

See `AGENTS.md` for the authoritative refactoring policy — it contains binding constraints (listed in "Invariants" below) and a priority-ordered task list (T1–T10 + factory pattern).

이 프로젝트의 완성도를 높이기 위해 추가 실험을 설계한다.

