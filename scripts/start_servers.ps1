# Starts the llama-server instances used by the main runs.
# The LLM gets the GPU to itself: the embedder runs on CPU (it only embeds short queries; the
# corpus index is precomputed) and reranker scores are precomputed by src/rerank_cache.py.
#   .\scripts\start_servers.ps1 -Model qwen38-27b
#   .\scripts\start_servers.ps1 -Model gemma4-12b -OnlyLLM   # swap the generator, keep the embedder
param(
    [ValidateSet("qwen38-27b", "gemma4-12b")] [string]$Model = "qwen38-27b",
    [string]$EmbedFile = "harrier-oss-v1-0.6b.Q8_0.gguf",
    [string]$EmbedPooling = "last",
    [int]$Parallel = 6,  # 6 slots fit in 16 GB of VRAM
    [int]$CtxPerSlot = 4096,
    [switch]$OnlyLLM
)
$root = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $root "tools\llama.cpp\llama-server.exe"
$models = Join-Path $root "models"
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force $logs | Out-Null

$files = @{ "qwen38-27b" = "Qwen3.8-27B-UD-Q3_K_XL.gguf"; "gemma4-12b" = "gemma-4-12b-it-Q8_0.gguf" }

Get-CimInstance Win32_Process -Filter "name='llama-server.exe'" |
    Where-Object { -not $OnlyLLM -or $_.CommandLine -match "--port 8080" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Seconds 3

$llmArgs = @("-m", (Join-Path $models $files[$Model]), "--port", "8080", "-ngl", "999", "-fa", "on",
             "-c", ($Parallel * $CtxPerSlot), "-np", $Parallel, "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
             "--jinja", "--reasoning-budget", "0", "--no-mmproj")
Start-Process $bin -ArgumentList $llmArgs -WindowStyle Hidden `
    -RedirectStandardOutput "$logs\llm.out.log" -RedirectStandardError "$logs\llm.err.log"

if (-not $OnlyLLM) {
    Start-Process $bin -ArgumentList @("-m", (Join-Path $models $EmbedFile), "--port", "8081", "--embedding",
        "--pooling", $EmbedPooling, "-ngl", "0", "-c", "4096", "-np", "4", "-t", "12") -WindowStyle Hidden `
        -RedirectStandardOutput "$logs\embed.out.log" -RedirectStandardError "$logs\embed.err.log"
}

foreach ($port in $(if ($OnlyLLM) { @(8080) } else { @(8080, 8081) })) {
    for ($i = 0; $i -lt 180; $i++) {
        try { if ((Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 2).status -eq "ok") { break } } catch {}
        Start-Sleep -Seconds 2
    }
    Write-Host "port $port ready"
}
