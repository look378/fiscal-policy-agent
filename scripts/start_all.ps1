# 一键启动：RAGFlow + embedding + 问答 + 预警（含监控栈）
# 用法：powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1 [-BgePath <模型目录>] [-SkipMonitoring]
param(
  [string]$BgePath = '',
  [switch]$SkipMonitoring
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root 'logs'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

function Test-Alive($port, $path, $name, $timeoutSec = 120) {
    # 探活：对需要鉴权的接口，401 也说明服务活着（404 则说明路径不对，应显式失败）
    $url = "http://localhost:$port$path"
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try { Invoke-WebRequest $url -TimeoutSec 3 -UseBasicParsing | Out-Null; Write-Host "  [OK] $name" -ForegroundColor Green; return $true }
        catch {
            $code = $_.Exception.Response.StatusCode.value__
            if ($code -eq 401 -or $code -eq 403) { Write-Host "  [OK] $name（$code 鉴权正常）" -ForegroundColor Green; return $true }
            Start-Sleep -Seconds 2
        }
    }
    Write-Host "  [FAIL] $name 未就绪" -ForegroundColor Red
    return $false
}

# bge-m3 权重目录：优先参数 → 环境变量 → 项目内 models/bge-m3 → 旧机器缓存目录（仅本机兜底）
if (-not $BgePath) {
  $cands = @(
    $env:BGE_M3_PATH,
    (Join-Path $Root 'models\bge-m3'),
    'D:\Project_Experience\finance_intelligent_agent\.modelscope-cache\models\BAAI--bge-m3\snapshots\master'
  )
  foreach ($c in $cands) {
    if ($c -and (Test-Path $c)) { $BgePath = $c; break }
  }
}
if (-not $BgePath -or -not (Test-Path $BgePath)) {
  Write-Host '[X] 找不到 bge-m3 模型目录，请用 -BgePath 指定（或设置环境变量 BGE_M3_PATH）' -ForegroundColor Red
  exit 1
}

$results = [ordered]@{}

function Wait-Http($url, $name, $timeoutSec = 120) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try { Invoke-RestMethod $url -TimeoutSec 3 | Out-Null; Write-Host "  [OK] $name" -ForegroundColor Green; return $true } catch { Start-Sleep -Seconds 2 }
    }
    Write-Host "  [FAIL] $name 未就绪" -ForegroundColor Red; return $false
}

function Test-PortFree($port) {
    return -not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

function Start-Svc($name, $script, $argList, $port, $healthUrl, $timeout, $logName) {
    if (-not (Test-PortFree $port)) {
        Write-Host "  [WARN] 端口 $port 已被占用，跳过启动 $name（可能是已在运行的实例）" -ForegroundColor Yellow
        $results[$name] = (Wait-Http $healthUrl $name 10)
        return
    }
    # 路径含空格时要加引号，否则 Start-Process 会拆成多个参数
    $quoted = '"' + (Resolve-Path $script).Path + '"'
    Start-Process -FilePath 'python' -ArgumentList (@($quoted) + $argList) -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogDir "$logName.log") `
        -RedirectStandardError (Join-Path $LogDir "$logName.err.log") | Out-Null
    $results[$name] = (Wait-Http $healthUrl $name $timeout)
}

function Invoke-ComposeUp($dir, $project) {
    # PS5.1 下原生命令的 stderr 在 EAP=Stop 时会抛 NativeCommandError，这里局部降为 Continue
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    Push-Location $dir
    try {
        docker compose -p $project up -d 2>&1 | Select-Object -Last 3
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
        $ErrorActionPreference = $old
    }
    return $code -eq 0
}

Write-Host '[1/5] 启动 RAGFlow 栈' -ForegroundColor Cyan
$ok = Invoke-ComposeUp (Join-Path $Root 'ragflow') 'fiscal-ragflow'
$results['RAGFlow'] = $ok -and (Test-Alive 9380 '/v1/system/version' 'RAGFlow API' 180)

Write-Host '[2/5] 启动本地 embedding 服务 (8900)' -ForegroundColor Cyan
Start-Svc 'embedding' (Join-Path $Root 'serving\embed_server.py') @('--model-path', $BgePath, '--port', '8900', '--name', 'bge-m3') 8900 'http://127.0.0.1:8900/health' 300 'embed'

Write-Host '[3/5] 启动问答服务 (8901)' -ForegroundColor Cyan
Start-Svc 'qa' (Join-Path $Root 'serving\qa_service.py') @('--port', '8901') 8901 'http://127.0.0.1:8901/health' 90 'qa'

Write-Host '[4/5] 启动预警服务 (8902)' -ForegroundColor Cyan
Start-Svc 'alert' (Join-Path $Root 'monitoring\alert_service\alert_service.py') @('--port', '8902', '--interval', '60') 8902 'http://127.0.0.1:8902/health' 90 'alert'

if (-not $SkipMonitoring) {
    Write-Host '[5/5] 启动监控栈（Prometheus + Grafana）' -ForegroundColor Cyan
    $null = Invoke-ComposeUp (Join-Path $Root 'monitoring') 'fiscal-monitoring'
    $results['prometheus'] = (Wait-Http 'http://localhost:19090/-/healthy' 'Prometheus' 120)
    $results['grafana'] = (Wait-Http 'http://localhost:13000/api/health' 'Grafana' 180)
} else {
    Write-Host '[5/5] 跳过监控栈（-SkipMonitoring）' -ForegroundColor Yellow
}

Write-Host ''
$failed = @($results.Keys | Where-Object { -not $results[$_] })
if ($failed.Count -gt 0) {
    Write-Host ('启动失败：' + ($failed -join ', ')) -ForegroundColor Red
    Write-Host "日志目录：$LogDir"
    exit 1
}

Write-Host '全部启动完成：' -ForegroundColor Green
Write-Host '  RAGFlow Web : http://localhost/'
Write-Host '  RAGFlow API : http://localhost:9380/api/v1'
Write-Host '  Dify        : http://localhost:8080（需在 dify015/ 下 docker compose up -d）'
Write-Host '  问答服务    : http://127.0.0.1:8901/api/ask'
Write-Host '  预警服务    : http://127.0.0.1:8902/alerts'
Write-Host '  Prometheus  : http://localhost:19090'
Write-Host '  Grafana     : http://localhost:13000'
