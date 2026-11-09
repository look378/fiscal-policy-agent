# 一键部署到 kind 集群
# 用法: powershell -ExecutionPolicy Bypass -File scripts\deploy_kind.ps1 [-ClusterName fiscal] [-Tag 1.0.0]
param([string]$ClusterName = 'fiscal', [string]$Tag = '1.0.0')

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Manifest = Join-Path $Root 'k8s\manifests.yaml'

Write-Host '[1/6] 检查工具链' -ForegroundColor Cyan
$kind = Get-Command kind -ErrorAction SilentlyContinue
if (-not $kind) {
  Write-Host '  [X] 未安装 kind。安装方式（Windows）：' -ForegroundColor Red
  Write-Host '      winget install Kubernetes.kind   或   choco install kind'
  Write-Host '      下载页: https://kind.sigs.k8s.io/docs/user/quick-start/#installation'
  exit 1
}
$kubectl = Get-Command kubectl -ErrorAction SilentlyContinue
if (-not $kubectl) { Write-Host '  [X] 未安装 kubectl' -ForegroundColor Red; exit 1 }
Write-Host "  kind: $($kind.Source)"

Write-Host '[2/6] 确保 kind 集群存在' -ForegroundColor Cyan
$clusters = (& kind get clusters) 2>$null
if ($clusters -notcontains $ClusterName) {
  Write-Host "  创建集群 $ClusterName ..."
  & kind create cluster --name $ClusterName
  if ($LASTEXITCODE -ne 0) { Write-Host '  [X] kind create cluster 失败' -ForegroundColor Red; exit 1 }
} else {
  Write-Host "  集群 $ClusterName 已存在"
}

Write-Host '[3/6] 构建镜像' -ForegroundColor Cyan
& (Join-Path $PSScriptRoot 'build_images.ps1') -Tag $Tag

Write-Host '[4/6] 加载镜像到 kind 节点' -ForegroundColor Cyan
& kind load docker-image "fiscal-qa:$Tag" --name $ClusterName
if ($LASTEXITCODE -ne 0) { Write-Host '  [X] 加载 fiscal-qa 镜像失败' -ForegroundColor Red; exit 1 }
& kind load docker-image "fiscal-alert:$Tag" --name $ClusterName
if ($LASTEXITCODE -ne 0) { Write-Host '  [X] 加载 fiscal-alert 镜像失败' -ForegroundColor Red; exit 1 }

Write-Host '[5/6] 检测宿主机在 kind 网络中的 IP' -ForegroundColor Cyan
$gw = (docker network inspect kind --format '{{(index .IPAM.Config 0).Gateway}}' 2>$null)
if (-not $gw) { Write-Host '  [X] 无法获取 kind 网络网关' -ForegroundColor Red; exit 1 }
Write-Host "  host gateway: $gw"

Write-Host '[6/6] 渲染并应用清单' -ForegroundColor Cyan
$rendered = Join-Path $env:TEMP ("fiscal-k8s-{0}.yaml" -f (Get-Date -Format 'HHmmss'))
$yaml = (Get-Content $Manifest -Raw -Encoding UTF8).
  Replace('__HOST_GATEWAY__', $gw).
  Replace('fiscal-qa:1.0.0', "fiscal-qa:$Tag").
  Replace('fiscal-alert:1.0.0', "fiscal-alert:$Tag")
# 不用 Set-Content -Encoding UTF8（PS5.1 会写 BOM，kubectl 处理 BOM 有问题）
[System.IO.File]::WriteAllText($rendered, $yaml, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "  镜像 tag: $Tag"
& kubectl apply -f $rendered
if ($LASTEXITCODE -ne 0) { Write-Host '  [X] kubectl apply 失败' -ForegroundColor Red; exit 1 }

Write-Host ''
Write-Host '等待 rollout ...' -ForegroundColor Cyan
& kubectl -n fiscal rollout status deployment/fiscal-qa --timeout=180s
$qaOk = ($LASTEXITCODE -eq 0)
& kubectl -n fiscal rollout status deployment/fiscal-alert --timeout=180s
$alertOk = ($LASTEXITCODE -eq 0)
Remove-Item $rendered -Force -ErrorAction SilentlyContinue
if (-not ($qaOk -and $alertOk)) {
    Write-Host '  [X] rollout 未完成（qa=' + $qaOk + ', alert=' + $alertOk + '），查看: kubectl -n fiscal describe pods' -ForegroundColor Red
    exit 1
}

Write-Host ''
Write-Host '部署结果：' -ForegroundColor Green
& kubectl -n fiscal get deploy,svc,pods
Write-Host ''
Write-Host '自测命令：' -ForegroundColor Green
Write-Host "  kubectl -n fiscal port-forward svc/fiscal-qa 8899:80"
Write-Host '  curl -X POST http://127.0.0.1:8899/api/ask -H "Content-Type: application/json" -d "{\"question\":\"预留份额比例是多少\"}"'
