# stemapp を Tailscale Serve で tailnet の中（自分の iPhone など）だけに公開する。
# Windows PowerShell 5.1 / PowerShell 7 両対応。このファイルは UTF-8（BOM 付き）で保存すること。
#
# 使い方（リポジトリ直下で）:
#   .\scripts\tailscale-serve.ps1 start    # 公開を始める（https://<この PC の名前>/ → http://127.0.0.1:<port>）
#   .\scripts\tailscale-serve.ps1 status   # 今の状態を表示する
#   .\scripts\tailscale-serve.ps1 stop     # 公開をやめる
#   -Port 8000 で転送先のポートを指定できる（省略時は .env の STEMAPP_PORT、無ければ 8000）。
#
# インターネット全体に公開する Funnel は使わない（このスクリプトは funnel を一切有効にしない）。
# TCP 転送（tailscale serve --tcp）や SSH のポート転送でも開かないこと
# （「保存フォルダを開く」を、この PC からの操作と取り違えないため。HTTPS の serve だけを使う）。

param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'status', 'stop')]
    [string]$Action = 'status',
    [int]$Port = 0
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot

function Write-Ok([string]$msg) { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "[注意] $msg" -ForegroundColor Yellow }
function Write-Ng([string]$msg) { Write-Host "[NG]   $msg" -ForegroundColor Red }

function Find-Tailscale {
    $cmd = Get-Command 'tailscale' -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $default = Join-Path $env:ProgramFiles 'Tailscale\tailscale.exe'
    if (Test-Path $default) { return $default }
    return $null
}

function Get-EnvValue([string]$name) {
    $envFile = Join-Path $Root '.env'
    if (-not (Test-Path $envFile)) { return $null }
    $line = Select-String -Path $envFile -Encoding UTF8 -Pattern "^\s*$name\s*=\s*(.*)$" |
        Select-Object -First 1
    if (-not $line) { return $null }
    return $line.Matches[0].Groups[1].Value.Trim().Trim('"', "'").Trim()
}

function Get-ServeJson([string]$ts) {
    $text = (& $ts serve status --json) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw 'tailscale serve status に失敗しました。' }
    if (-not $text.Trim()) { return $null }
    return $text | ConvertFrom-Json
}

function Test-FunnelOn($serve) {
    if ($null -eq $serve -or $null -eq $serve.AllowFunnel) { return $false }
    foreach ($p in $serve.AllowFunnel.PSObject.Properties) {
        if ($p.Value -eq $true) { return $true }
    }
    return $false
}

function Show-Status([string]$ts, [string]$dnsName) {
    Write-Host ''
    Write-Host '--- tailscale serve status ---'
    & $ts serve status
    Write-Host '------------------------------'
    $serve = Get-ServeJson $ts
    if (Test-FunnelOn $serve) {
        Write-Ng 'Funnel（インターネット全体への公開）が有効です。stemapp では使いません。'
        Write-Host "      止めるには: & '$ts' funnel reset  （または Tailscale の管理画面で無効にする）"
    }
    else {
        Write-Ok 'Funnel は使っていません（公開は tailnet の中だけ）'
    }
    $allowed = Get-EnvValue 'STEMAPP_ALLOWED_HOSTS'
    if ($env:STEMAPP_ALLOWED_HOSTS) { $allowed = $env:STEMAPP_ALLOWED_HOSTS }
    if ($dnsName) {
        $names = @()
        if ($allowed) { $names = $allowed.Split(',') | ForEach-Object { $_.Trim().ToLower() } }
        if ($names -contains $dnsName.ToLower()) {
            Write-Ok "設定 STEMAPP_ALLOWED_HOSTS に $dnsName が入っています"
        }
        else {
            Write-Warn ".env に次の1行を足して stemapp を起動し直してください（無いと 400 になります）:"
            Write-Host "      STEMAPP_ALLOWED_HOSTS=$dnsName"
        }
    }
}

try {
    $ts = Find-Tailscale
    if (-not $ts) { throw 'Tailscale が見つかりません。https://tailscale.com/download から入れてください。' }

    $status = ((& $ts status --json) -join "`n") | ConvertFrom-Json
    if ($status.BackendState -ne 'Running') {
        throw "Tailscale にログインしていないか停止中です（$($status.BackendState)）。タスクトレイから接続してください。"
    }
    $dnsName = "$($status.Self.DNSName)".TrimEnd('.')

    if ($Port -le 0) {
        $fromEnv = Get-EnvValue 'STEMAPP_PORT'
        $Port = if ($fromEnv) { [int]$fromEnv } else { 8000 }
    }
    $target = "http://127.0.0.1:$Port"

    switch ($Action) {
        'start' {
            Write-Host "tailnet の中だけに公開します: https://$dnsName/ → $target"
            & $ts serve --bg --https=443 $target
            if ($LASTEXITCODE -ne 0) { throw 'tailscale serve に失敗しました。' }
            Show-Status $ts $dnsName
            Write-Host ''
            Write-Host "iPhone（同じ tailnet に参加済み）の Safari で https://$dnsName/ を開いてください。"
            Write-Host 'stemapp が起動していないと 502 になります（scripts\start.bat で起動）。'
        }
        'stop' {
            & $ts serve --https=443 off
            if ($LASTEXITCODE -ne 0) { Write-Warn '止める対象が無かったか、止められませんでした。' }
            else { Write-Ok '公開をやめました。' }
            Show-Status $ts $dnsName
        }
        default {
            Write-Host "この PC の名前: $dnsName（転送先の想定: $target）"
            Show-Status $ts $dnsName
        }
    }
}
catch {
    Write-Ng $_.Exception.Message
    exit 1
}
