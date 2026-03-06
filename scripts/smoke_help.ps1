#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

$commands = @(
    @("python", "pipeline/main.py", "--help"),
    @("python", "src/build_vectordb_search.py", "--help"),
    @("python", "src/retrieval_system/main.py", "--help"),
    @("python", "src/preprocess_and_generate_answer.py", "--help"),
    @("python", "pipeline/run_pipeline.py", "--help")
)

foreach ($cmd in $commands) {
    Write-Host ("Running: " + ($cmd -join " "))
    & $cmd[0] $cmd[1] $cmd[2]
    if ($LASTEXITCODE -ne 0) {
        throw ("Smoke help check failed: " + ($cmd -join " "))
    }
}

Write-Host "Smoke help checks passed."
