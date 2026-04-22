#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

$PythonExe = "python"
$VenvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
}

$commands = @(
    @($PythonExe, "-m", "pipeline.run_pipeline", "--help"),
    @($PythonExe, "-m", "pipeline.step1_search", "--help"),
    @($PythonExe, "-m", "pipeline.step2_decompose", "--help"),
    @($PythonExe, "-m", "pipeline.step3_build_vectordb", "--help"),
    @($PythonExe, "-m", "pipeline.step4_retrieve", "--help"),
    @($PythonExe, "-m", "pipeline.step5_generate", "--help"),
    @($PythonExe, "-m", "pipeline.evaluate.eval_search", "--help"),
    @($PythonExe, "-m", "pipeline.evaluate.eval_answers", "--help"),
    @($PythonExe, "-m", "src.build_vectordb_search", "--help"),
    @($PythonExe, "-m", "src.retrieval_system.main", "--help"),
    @($PythonExe, "-m", "src.preprocess_and_generate_answer", "--help"),
    @($PythonExe, "-m", "src.multi_hop_to_single_hop", "--help")
)

foreach ($cmd in $commands) {
    Write-Host ("Running: " + ($cmd -join " "))
    $exe = $cmd[0]
    $args = $cmd[1..($cmd.Length - 1)]
    & $exe @args
    if ($LASTEXITCODE -ne 0) {
        throw ("Smoke help check failed: " + ($cmd -join " "))
    }
}

Write-Host "Smoke help checks passed."
