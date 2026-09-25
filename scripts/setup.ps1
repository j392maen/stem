# stemapp セットアップ（Windows PowerShell 5.1 / PowerShell 7 両対応）
# 使い方: リポジトリ直下で  .\scripts\setup.ps1
# 何度実行しても壊れない。既にあるものはスキップする。
# 注意: このファイルは UTF-8（BOM 付き）で保存すること（PowerShell 5.1 で日本語を正しく読むため）。

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Write-Step([string]$msg) { Write-Host "`n== $msg ==" -ForegroundColor Cyan }
function Write-Ok([string]$msg) { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "[注意] $msg" -ForegroundColor Yellow }
function Write-Ng([string]$msg) { Write-Host "[NG]   $msg" -ForegroundColor Red }

function Update-PathFromRegistry {
    # winget で入れた直後でも、このシェルでコマンドが使えるようにする
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

function Test-Command([string]$name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Install-IfMissing([string]$command, [string]$wingetId, [string]$label) {
    if (Test-Command $command) {
        Write-Ok "$label は導入済み"
        return
    }
    if (-not (Test-Command 'winget')) {
        throw "$label が無く、winget も使えません。$label を手動で導入してください。"
    }
    Write-Host "[導入] $label を winget で入れます（$wingetId）"
    winget install --id $wingetId -e --accept-source-agreements --accept-package-agreements --silent
    Update-PathFromRegistry
    if (-not (Test-Command $command)) {
        throw "$label を入れましたが見つかりません。新しい PowerShell を開いて再実行してください。"
    }
    Write-Ok "$label を導入しました"
}

function Invoke-Checked([string]$label, [scriptblock]$block) {
    & $block
    if ($LASTEXITCODE -ne 0) { throw "$label に失敗しました（終了コード $LASTEXITCODE）" }
}

try {
    Update-PathFromRegistry

    Write-Step '1. 道具の確認'
    if (Test-Command 'git') { Write-Ok "git: $(git --version)" }
    else { Write-Warn 'git が見つかりません（開発に使う場合は Git for Windows を入れてください）' }
    Install-IfMissing 'uv' 'astral-sh.uv' 'uv'
    Install-IfMissing 'ffmpeg' 'Gyan.FFmpeg' 'ffmpeg'
    Install-IfMissing 'deno' 'DenoLand.Deno' 'Deno'
    if (Test-Command 'nvidia-smi') {
        $gpu = (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) -join ', '
        Write-Ok "GPU: $gpu"
    }
    else { Write-Warn 'nvidia-smi が見つかりません（NVIDIA ドライバを確認してください。CPU でも動きます）' }

    Write-Step '2. Python 環境（uv sync）'
    Invoke-Checked 'uv sync' { uv sync --extra gpu --extra url }
    Write-Ok '依存パッケージを揃えました'

    Write-Step '3. 設定ファイル（.env）'
    $envFile = Join-Path $Root '.env'
    if (Test-Path $envFile) {
        Write-Ok '.env は既にあります（変更しません）'
    }
    else {
        Copy-Item (Join-Path $Root '.env.example') $envFile
        Write-Ok '.env.example から .env を作りました'
    }

    # yt-dlp.exe の確認（.env の STEMAPP_YTDLP_PATH、無ければ既定の場所）
    $ytdlp = 'C:\mine\yt-dlp.exe'
    $line = Select-String -Path $envFile -Pattern '^\s*STEMAPP_YTDLP_PATH\s*=\s*(.+)$' |
        Select-Object -First 1
    if ($line) { $ytdlp = $line.Matches[0].Groups[1].Value.Trim().Trim('"', "'").Trim() }
    if (Test-Path $ytdlp) { Write-Ok "yt-dlp: $ytdlp（$(& $ytdlp --version)）" }
    else { Write-Warn "yt-dlp.exe が $ytdlp にありません（Python 版 yt-dlp で代用します）" }

    Write-Step '4. データベース'
    Invoke-Checked 'stemapp init-db' { uv run stemapp init-db }

    Write-Step '5. 環境診断'
    uv run stemapp doctor
    if ($LASTEXITCODE -ne 0) {
        Write-Ng '診断で NG がありました。表の「対処のヒント」を確認してください。'
        exit 1
    }
    Write-Host "`nセットアップ完了。起動は scripts\start.bat（または uv run stemapp serve）。" -ForegroundColor Green
}
catch {
    Write-Ng $_.Exception.Message
    exit 1
}
