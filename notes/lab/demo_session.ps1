# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0
#
# 시연 세션 — 화면 녹화용으로 실제 명령을 순서대로 실행한다.
# 명령은 타이핑하듯 한 글자씩 찍고, 출력은 실제 실행 결과 그대로 보여 준다.

$ErrorActionPreference = "Continue"
# 콘솔 코드페이지를 UTF-8로 바꾸지 않으면 한글 출력이 깨진다.
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
# 녹화 대상 창을 특정하고, 글자가 크게 보이도록 열·행 수를 줄인다.
$Host.UI.RawUI.WindowTitle = "routerx demo"
try {
    $Host.UI.RawUI.BufferSize = New-Object Management.Automation.Host.Size(110, 3000)
    $Host.UI.RawUI.WindowSize  = New-Object Management.Automation.Host.Size(110, 32)
} catch {}
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "src"
Set-Location "d:\opensource\ossp-2026-llm-router-challenge"

function Show-Prompt {
    Write-Host ""
    Write-Host "PS " -NoNewline -ForegroundColor DarkGray
    Write-Host "routerx" -NoNewline -ForegroundColor Cyan
    Write-Host " > " -NoNewline -ForegroundColor DarkGray
}

function Type-Line($text, $delay = 18) {
    foreach ($ch in $text.ToCharArray()) {
        Write-Host $ch -NoNewline -ForegroundColor White
        Start-Sleep -Milliseconds $delay
    }
    Write-Host ""
}

function Section($text) {
    Write-Host ""
    Write-Host ("  " + $text) -ForegroundColor DarkCyan
    Write-Host ("  " + ("-" * 66)) -ForegroundColor DarkGray
    Start-Sleep -Milliseconds 700
}

Clear-Host
Write-Host ""
Write-Host "  프롬프트 난이도 인지 기반 경량 LLM 라우터" -ForegroundColor White
Write-Host "  팀 트리아지 · SK텔레콤 지정과제" -ForegroundColor DarkGray
Start-Sleep -Seconds 2

# 1. 라우팅 실행
Section "1. 라우팅 — 프롬프트만 보고 모델 선택"
Show-Prompt
Type-Line "python -m routerx.cli --input data/materialized/dev/inputs.json --tier premium --output build/demo/premium.json"
python -m routerx.cli --input data/materialized/dev/inputs.json --tier premium --output build/demo/premium.json
Start-Sleep -Milliseconds 1200

Show-Prompt
Type-Line "python build/demo/show_pick.py"
python build/demo/show_pick.py
Start-Sleep -Seconds 2

# 2. 공식 채점기
Section "2. 공식 채점기로 검증"
Show-Prompt
Type-Line "python -m ossp_router.cli self-check --submissions build/demo/sub --report build/demo/report.json"
foreach ($t in @("fast","balanced","premium")) {
    python -m routerx.cli --input data/materialized/dev/inputs.json --tier $t --output "build/demo/sub/$t.json" | Out-Null
}
python -m ossp_router.cli self-check --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json --submissions build/demo/sub --report build/demo/report.json
python build/demo/show_result.py
Start-Sleep -Seconds 3

# 3. 자체 점검
Section "3. 자체 점검 — 규칙 준수·성능·예산"
Show-Prompt
Type-Line "python -m routerx.audit --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json"
python -m routerx.audit --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json
Start-Sleep -Seconds 3

# 4. 컨테이너
Section "4. 제출 이미지 — 공식 자원 한도로 실행"
Show-Prompt
Type-Line "docker run --platform linux/arm64 --network none --read-only --cpus 2 --memory 2g ghcr.io/leelang7/routerx@sha256:ea01be4a..."
docker run --rm --platform linux/arm64 --network none --read-only --user 65532:65532 --cpus 2 --memory 2g --memory-swap 2g --pids-limit 32 --tmpfs /tmp:size=256m -v "${PWD}\data\materialized\dev:/challenge/input:ro" -v "${PWD}\build\demo\ctr:/challenge/output" ghcr.io/leelang7/routerx@sha256:ea01be4aa373f1358450c56105f4f595619b7fa2bd272d418c9bc71f8b75016f --input /challenge/input/inputs.json --tier fast --output /challenge/output/submission.json
Write-Host ""
Write-Host "  네트워크 없음 · 읽기 전용 루트 · CPU 2코어 · 메모리 2GiB" -ForegroundColor DarkGray
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "  시연 종료" -ForegroundColor DarkCyan
Start-Sleep -Seconds 2
