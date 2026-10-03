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
#
# tailscale の出力は UTF-8。PowerShell 5.1 はコンソールの文字コード（日本語の Windows は cp932）で
# 読むため、表示名などの日本語が化けて JSON が壊れる。そこで出力をバイト列のまま受けて UTF-8 で
# 解釈し（Invoke-Tailscale）、JSON からは必要な項目（BackendState・DNSName・AllowFunnel）だけを読む。

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

# tailscale を実行し、標準出力・標準エラーを UTF-8 として読む（コンソールの文字コードに依らない）
function Invoke-Tailscale([string]$ts, [string[]]$tsArgs) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $ts
    # 引数は英数字と記号だけ（空白を含まない）なので、空白でつなぐだけでよい
    $psi.Arguments = ($tsArgs -join ' ')
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    $psi.StandardOutputEncoding = $utf8
    $psi.StandardErrorEncoding = $utf8
    $proc = [System.Diagnostics.Process]::Start($psi)
    $outTask = $proc.StandardOutput.ReadToEndAsync()
    $err = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
    return [pscustomobject]@{ Code = $proc.ExitCode; Out = $outTask.Result; Err = $err }
}

function Write-TsOutput($res) {
    foreach ($text in @($res.Out, $res.Err)) {
        if ($text -and $text.Trim()) { Write-Host $text.TrimEnd() }
    }
}

# JSON の中の最初の "名前": "値" の値（文字列）。無ければ $null
function Get-JsonString([string]$json, [string]$name) {
    $m = [regex]::Match($json, '"' + [regex]::Escape($name) + '"\s*:\s*"((?:[^"\\]|\\.)*)"')
    if ($m.Success) { return $m.Groups[1].Value }
    return $null
}

# tailscale status --json から必要な項目だけを読む（表示名などの他の項目は読まない）
function Get-TsStatus([string]$ts) {
    $res = Invoke-Tailscale $ts @('status', '--json')
    $json = "$($res.Out)"
    $state = Get-JsonString $json 'BackendState'
    if (-not $state) {
        $detail = "$($res.Err)$($res.Out)".Trim()
        throw "tailscale status を読めません（Tailscale は起動していますか）。$detail"
    }
    # 自分（Self）の DNSName。Self の中で DNSName より前にある項目は入れ子を持たない
    $dns = ''
    $m = [regex]::Match($json, '"Self"\s*:\s*\{[^{}]*?"DNSName"\s*:\s*"([^"]*)"')
    if ($m.Success) { $dns = $m.Groups[1].Value }
    return [pscustomobject]@{ BackendState = $state; DNSName = $dns.TrimEnd('.') }
}

# serve の設定で Funnel（インターネット全体への公開）が有効か。
# AllowFunnel は {"<名前>:443": true} の形（入れ子なし）
function Test-FunnelOn([string]$ts) {
    $res = Invoke-Tailscale $ts @('serve', 'status', '--json')
    if ($res.Code -ne 0) { throw 'tailscale serve status に失敗しました。' }
    $m = [regex]::Match("$($res.Out)", '"AllowFunnel"\s*:\s*\{([^{}]*)\}')
    return $m.Success -and ($m.Groups[1].Value -match ':\s*true\b')
}

function Show-Status([string]$ts, [string]$dnsName) {
    Write-Host ''
    Write-Host '--- tailscale serve status ---'
    Write-TsOutput (Invoke-Tailscale $ts @('serve', 'status'))
    Write-Host '------------------------------'
    if (Test-FunnelOn $ts) {
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

    $status = Get-TsStatus $ts
    if ($status.BackendState -ne 'Running') {
        throw "Tailscale にログインしていないか停止中です（$($status.BackendState)）。タスクトレイから接続してください。"
    }
    $dnsName = $status.DNSName

    if ($Port -le 0) {
        $fromEnv = Get-EnvValue 'STEMAPP_PORT'
        $Port = if ($fromEnv) { [int]$fromEnv } else { 8000 }
    }
    $target = "http://127.0.0.1:$Port"

    switch ($Action) {
        'start' {
            Write-Host "tailnet の中だけに公開します: https://$dnsName/ → $target"
            $res = Invoke-Tailscale $ts @('serve', '--bg', '--https=443', $target)
            Write-TsOutput $res
            if ($res.Code -ne 0) { throw 'tailscale serve に失敗しました。' }
            Show-Status $ts $dnsName
            Write-Host ''
            Write-Host "iPhone（同じ tailnet に参加済み）の Safari で https://$dnsName/ を開いてください。"
            Write-Host 'stemapp が起動していないと 502 になります（scripts\start.bat で起動）。'
        }
        'stop' {
            $res = Invoke-Tailscale $ts @('serve', '--https=443', 'off')
            Write-TsOutput $res
            if ($res.Code -ne 0) { Write-Warn '止める対象が無かったか、止められませんでした。' }
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
