# 构建本项目的两个镜像（构建上下文 = 仓库根目录）
# 用法: powershell -ExecutionPolicy Bypass -File scripts\build_images.ps1 [-Tag 1.0.0]
param([string]$Tag = '1.0.0')

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host '[0/2] 检查 Docker daemon' -ForegroundColor Cyan
docker info --format '{{.ServerVersion}}' 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host '  [X] Docker daemon 未运行（先启动 Docker Desktop）' -ForegroundColor Red
    exit 1
}

Write-Host "[1/2] 构建 fiscal-qa:$Tag" -ForegroundColor Cyan
docker build -f serving/Dockerfile -t "fiscal-qa:$Tag" .
if ($LASTEXITCODE -ne 0) { Write-Host '  [X] fiscal-qa 构建失败' -ForegroundColor Red; exit 1 }

Write-Host "[2/2] 构建 fiscal-alert:$Tag" -ForegroundColor Cyan
docker build -f monitoring/alert_service/Dockerfile -t "fiscal-alert:$Tag" .
if ($LASTEXITCODE -ne 0) { Write-Host '  [X] fiscal-alert 构建失败' -ForegroundColor Red; exit 1 }

Write-Host '构建完成：' -ForegroundColor Green
docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | Select-String 'fiscal'
if ($LASTEXITCODE -ne 0) { exit 1 }
