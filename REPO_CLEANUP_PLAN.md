# 프로페셔널 GitHub 레포지토리 정리 계획서

> Target: SHRAG (Search like Human with RAG)
> Scope: 코드/문서/도구 체인을 OSS 표준에 맞게 재구조화
> Date: 2026-04-30

---

## 1. 현황 진단

### 1.1 루트 디렉토리 혼잡 (35+ 항목)
- **메타 문서 과잉 (7개)**: `EXPERIMENT_BLUEPRINT.md`, `EXPERIMENT_MANUAL.md`, `PROJECT_IMPROVEMENT_PLAN.md`, `VERIFY_REPORT.md`, `ANALYSIS.md`, `AGENTS.md`, `CLAUDE.md`
- **개인용 한국어 파일 4개 루트 노출**: `이력서 용 프로젝트 요약.md`, `프로젝트 요약 첨삭본.md`, `프로젝트 요약.md`, `프로젝트기술서.md`
- **대형 바이너리 추적**: `test.zip` (2.8 MB), `shrag.pdf` (2.5 MB)
- **임시/깨진 산출물**: `tmp_verify/`, `pytest-cache-files-*` 6개(권한 `d--x--x--x` 손상)
- **표준 OSS 파일 부재**: `LICENSE`, `CONTRIBUTING.md`, `CHANGELOG.md`, `.github/` 없음

### 1.2 코드 구조 문제
- **이중 import 루트**: `src/` (대형, 혼잡) + `pipeline/` (정돈) → 책임 분리 모호
- **`src/` 내부 혼잡**: 모듈/스크립트/노트북(`main.ipynb` 244 KB) 혼재, 30+ 최상위 파일
- **legacy 코드 잔존**: `pipeline/evaluate/legacy/`, `src/search_pipeline/legacy/`
- **`pipeline/.DS_Store` 추적됨**

### 1.3 패키징/품질 도구 부재
- `pyproject.toml`/`setup.py` 없음 → `pip install -e .` 불가
- 린터/포매터/타입체커 미설정 (ruff, mypy, black 등)
- 루트 `tests/` 부재, 테스트는 `experiments/tests/`에 2개 파일만
- pre-commit, CI/CD(`.github/workflows/`) 없음

### 1.4 산출물/데이터 위생
- `outputs/` 30+ 캐시 디렉토리(`_target_sweep_cache_v1prompt`, `v1real`, `v2`, `v3`, `v4`, `v4_t50` …) 누적
- `results/` legacy 폴더만 다수 (`pubmedqa_*`, `scifact_*`)
- `.gitignore`는 `data/`, `outputs/`, `results/` 무시하지만 일부 placeholder 추적 중

### 1.5 Git/브랜치 정책
- 메인 브랜치가 비표준 이름 `gpt-oss` (PR 베이스 혼란)
- 활성 브랜치 다수 (`new-experiment`, `refactor`, …) 미정리
- 커밋 메시지 일관성 부족 ("UPDATE", "pre-refactoring", 한글/영문 혼재)
- branch protection 규칙 없음

---

## 2. 목표 (정량)

| 항목 | 현재 | 목표 |
|---|---|---|
| 루트 항목 수 | 35+ | ≤ 12 |
| 추적 파일 수 | 179 | ~120 (legacy/대형 제거) |
| 메타 문서 (루트) | 7 | 1 (README) + `docs/` |
| import 루트 | 2 (`src/`, `pipeline/`) | 1 (`shrag/`) |
| 패키지 설치 | 불가 | `pip install -e .` |
| CI 상태 | 없음 | green badge |
| Python 진입점 | 흩어짐 | console_scripts 통합 |

---

## 3. 목표 디렉토리 구조

```
scion-rag-challenge/
├── shrag/                       # 단일 import 루트 (src/ + pipeline/ 통합)
│   ├── __init__.py
│   ├── cli.py                   # console_scripts 엔트리
│   ├── pipeline/                # 기존 pipeline/ 5단계
│   │   ├── run.py               # run_pipeline.py
│   │   └── steps/               # step1~step5
│   ├── retrieval/               # 기존 src/retrieval_system
│   ├── search/                  # 기존 src/search_pipeline (legacy 제거)
│   ├── evaluation/              # 기존 src/evaluate + pipeline/evaluate
│   ├── llm/                     # 기존 src/llm_client
│   ├── prompts/
│   ├── features/
│   └── data_handler/
├── experiments/                 # 그대로 유지 (이미 정돈됨)
├── tests/                       # pytest 루트 (신규)
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── configs/                     # 그대로 유지
├── data/                        # .gitignore (샘플만 `data/samples/` 추적)
├── docs/                        # 메타 문서 통합
│   ├── architecture.md          # README의 상세 버전
│   ├── experiments.md           # EXPERIMENT_BLUEPRINT 통합 요약
│   ├── manual.md                # EXPERIMENT_MANUAL 정리
│   ├── analysis/                # ANALYSIS.md, VERIFY_REPORT.md
│   └── notes/                   # 한국어 메모(.gitignore 권장)
├── scripts/                     # 셸 스크립트 (.ps1는 보존 결정 필요)
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   └── release.yml
│   ├── ISSUE_TEMPLATE/
│   ├── pull_request_template.md
│   ├── CODEOWNERS
│   └── dependabot.yml
├── .devcontainer/               # 그대로 유지
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml               # 신규 (PEP 621)
├── README.md                    # 재작성 (badges, quickstart)
├── LICENSE                      # 신규 (MIT 또는 Apache-2.0)
├── CONTRIBUTING.md              # 신규
├── CHANGELOG.md                 # 신규 (Keep a Changelog)
├── .gitignore                   # 보강
├── .env.example
└── .pre-commit-config.yaml      # 신규
```

---

## 4. Phase별 작업

### Phase 0 — 환경 위생 (0.5일)

| ID | 작업 | 위험도 |
|---|---|---|
| P0-1 | `pytest-cache-files-*` 6개 디렉토리 권한 복구 후 삭제 | 낮음 |
| P0-2 | `pipeline/.DS_Store` 추적 해제 + 전역 `.DS_Store` 무시 | 낮음 |
| P0-3 | `test.zip`, `shrag.pdf` Git 히스토리에서 제거 (`git filter-repo`) 또는 LFS 이전 | 중간 (히스토리 변경) |
| P0-4 | `tmp_verify/` 삭제 | 낮음 |
| P0-5 | 한국어 개인 파일 4개 → `docs/notes/` 이동 + `.gitignore` 추가 | 낮음 |

### Phase 1 — 코드 재구조화 (2-3일)

| ID | 작업 |
|---|---|
| P1-1 | `shrag/` 패키지 루트 생성, `src/`·`pipeline/` 통합 마이그레이션 |
| P1-2 | import 경로 일괄 치환 (스크립트화) — `from src.x` → `from shrag.x` |
| P1-3 | `src/main.ipynb`, `src/final_result*.py` 등 ad-hoc 스크립트 → `notebooks/` 또는 `scripts/legacy/` 분리 |
| P1-4 | `pipeline/evaluate/legacy/`, `src/search_pipeline/legacy/` 삭제 또는 `archive/` 격리 |
| P1-5 | `pipeline/run_pipeline.py`의 CLI를 `shrag/cli.py`로 노출, `console_scripts`에 등록 |
| P1-6 | 모든 `__init__.py` 정리, public API 선언 (`__all__`) |

> **Invariant**: AGENTS.md의 리팩토링 정책을 따른다. 각 모듈 이동 시 단위 테스트로 동등성 확인.

### Phase 2 — 패키징 (0.5일)

| ID | 작업 |
|---|---|
| P2-1 | `pyproject.toml` 작성 (PEP 621, build-backend: setuptools 또는 hatchling) |
| P2-2 | `requirements.txt` → `[project.dependencies]` 이전, 개발 의존은 `[project.optional-dependencies.dev]` |
| P2-3 | `console_scripts`: `shrag = shrag.cli:main`, `shrag-search`, `shrag-eval` |
| P2-4 | `pip install -e ".[dev]"` 동작 검증 |

### Phase 3 — 품질 도구 (0.5일)

| ID | 도구 | 설정 |
|---|---|---|
| P3-1 | ruff | lint + format 통합 (`pyproject.toml` 내 `[tool.ruff]`) |
| P3-2 | mypy | 점진적 (`strict = false`, 모듈별 `disallow_untyped_defs` 단계 적용) |
| P3-3 | pytest | `[tool.pytest.ini_options]` + `tests/` 루트 표준화 |
| P3-4 | coverage | `pytest-cov`, 임계 50% 시작 → 70% 목표 |
| P3-5 | pre-commit | ruff, mypy, end-of-file-fixer, trailing-whitespace, check-added-large-files |

### Phase 4 — CI/CD (0.5일)

| ID | 작업 |
|---|---|
| P4-1 | `.github/workflows/ci.yml`: lint → typecheck → test (Python 3.11, 3.12 매트릭스) |
| P4-2 | PR 템플릿, 이슈 템플릿(bug/feature) |
| P4-3 | `CODEOWNERS`, `dependabot.yml` (pip + github-actions weekly) |
| P4-4 | branch protection: `main` push 금지, PR 1 review + CI green 강제 |
| P4-5 | (선택) Docker 빌드 워크플로 |

### Phase 5 — 문서 (1일)

| ID | 작업 |
|---|---|
| P5-1 | `LICENSE` 결정 후 추가 (권장: Apache-2.0 — patent grant 포함) |
| P5-2 | `README.md` 재작성: badges(CI, license, python), 1-paragraph hero, quickstart 5줄, architecture 다이어그램, links |
| P5-3 | `CONTRIBUTING.md`: 개발 환경 셋업, 커밋 컨벤션(Conventional Commits), PR 절차 |
| P5-4 | `CHANGELOG.md`: Keep a Changelog 포맷, 첫 항목 `[Unreleased]` |
| P5-5 | `docs/` 통합: 7개 루트 문서 → `architecture.md`, `experiments.md`, `manual.md`, `analysis/` 4개로 압축 |
| P5-6 | (선택) MkDocs 또는 Sphinx 사이트 + GitHub Pages |

### Phase 6 — Git/브랜치 정책 (0.5일)

| ID | 작업 |
|---|---|
| P6-1 | 기본 브랜치 `gpt-oss` → `main` 변경 (GitHub UI에서 rename, 모든 ref 자동 갱신) |
| P6-2 | 활성 브랜치 정리: `new-experiment`, `refactor` 등 머지/삭제 결정 |
| P6-3 | branch protection: `main` 보호, force push 금지 |
| P6-4 | 커밋 컨벤션: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:` 도입 |
| P6-5 | (선택) release-please 또는 semantic-release로 태그/릴리스 자동화 |

---

## 5. 우선순위

| 등급 | Phase | 목표 |
|---|---|---|
| **P0 (Day 1)** | Phase 0, Phase 5 (LICENSE만) | 위생 + 라이선스 — OSS 가시성의 최소 조건 |
| **P1 (Week 1)** | Phase 1, Phase 2 | 단일 패키지 루트 + 표준 빌드 — 가장 큰 가치 |
| **P2 (Week 2)** | Phase 3, Phase 4 | 품질 자동화 — 회귀 방지 |
| **P3 (Week 2-3)** | Phase 5 (문서 통합), Phase 6 | 외부 협업자 온보딩 |
| **P4 (선택)** | Docker 빌드, MkDocs, PyPI 배포 | 성숙도 향상 |

---

## 6. 검증 체크리스트 (Definition of Done)

- [ ] `pip install -e ".[dev]"` 깨끗이 성공
- [ ] `pytest` 전체 통과 (기존 테스트 + 새 마이그레이션 단위 테스트)
- [ ] `ruff check .` + `ruff format --check .` 통과
- [ ] `mypy shrag/` 새 코드 무경고 (legacy 제외)
- [ ] CI 워크플로 그린 배지
- [ ] 루트 항목 ≤ 12개
- [ ] `python -c "import shrag"` 성공
- [ ] `shrag --help` 출력 (entry point 동작)
- [ ] README quickstart로 새 사용자 5분 안에 첫 쿼리 실행 가능
- [ ] LICENSE, CONTRIBUTING, CHANGELOG 존재
- [ ] `.github/workflows/ci.yml` 존재, branch protection 설정

---

## 7. 위험 요소 및 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| Phase 1 import 경로 변경으로 회귀 발생 | 높음 | 마이그레이션 전후 동일 입력→출력 골든 테스트 작성 |
| Git 히스토리 재작성(`test.zip` 제거) | 중간 | 별도 브랜치에서 `git filter-repo` 후 팀 공지, 강제 푸시 일정 합의 |
| 한국어 메모 .gitignore 시 협업자 혼란 | 낮음 | `docs/notes/README.md`에 정책 명시 |
| 기본 브랜치 변경으로 외부 PR 영향 | 낮음 | GitHub auto-redirect 활용, 30일 deprecation 공지 |

---

## 8. 1차 실행 권장 순서 (즉시 시작 가능)

1. **오늘**: Phase 0 전체 + LICENSE 결정/추가 (1시간)
2. **D+1**: `pyproject.toml` 초안(P2-1) — 구조 변경 없이 현재 `src/`+`pipeline/` 그대로 패키징
3. **D+2**: ruff + pre-commit + 최소 CI(`ci.yml`) 도입 — 이후 작업의 안전망
4. **D+3 ~ D+5**: Phase 1 코드 재구조화 (가장 큰 작업, 별도 PR 분할)
5. **D+6**: 문서/브랜치 정책 마무리

---

## 9. 참고: 다른 메타 문서와의 관계

- `EXPERIMENT_BLUEPRINT.md`, `PROJECT_IMPROVEMENT_PLAN.md` → **연구/실험 계획**
- `AGENTS.md` → **AI 에이전트 작업 지침**
- 본 문서 → **레포지토리 엔지니어링 위생 계획**

세 문서는 서로 직교하며, 본 정리 작업이 완료되면 위 두 문서도 `docs/` 하위로 통합한다 (Phase 5-5).
