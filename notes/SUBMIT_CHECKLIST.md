<!--
SPDX-FileCopyrightText: Copyright 2026 routerx contributors
SPDX-License-Identifier: Apache-2.0
-->

# 최종 제출 절차 (마감 2026-08-27 목 18:00)

코드가 확정된 뒤 아래를 **순서대로** 수행한다. 순서가 중요하다 —
이미지는 코드 커밋에서 빌드하고, 기술 정보 파일은 그 다음 커밋에 넣는다.

## 1. 코드 확정과 최종 점검

```console
# 아티팩트 재학습 (최종 정책 값으로)
PYTHONPATH=src python3 train_routerx/train.py \
  --train-input data/materialized/train/inputs.json \
  --train-outcomes data/train/outcomes.json \
  --dev-input data/materialized/dev/inputs.json \
  --dev-outcomes data/dev/outcomes.json \
  --artifact src/routerx/artifact.npz \
  --fit-on train+dev \
  <최종 확정된 --tier-margin / --tier-k1-cap / --k1-item-cap>

# 자체 점검 — 실패가 하나라도 있으면 제출하지 않는다
PYTHONPATH=src python3 -m routerx.audit \
  --input data/materialized/train/inputs.json --outcomes data/train/outcomes.json
PYTHONPATH=src python3 -m routerx.audit \
  --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json

# 전체 시험 (공식 저장소 시험 포함)
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

## 2. 코드 커밋 (이미지의 기준이 된다)

```console
git add -A && git commit -m "..."
git push origin routerx
git rev-parse HEAD          # ← 이 40자리가 commit_sha
```

## 3. arm64 이미지 빌드와 push

기반 이미지 다이제스트에 `linux/arm64`가 있는지 먼저 확인한다(없으면 빌드 실패).

```console
docker buildx imagetools inspect \
  python:3.11-slim-bookworm@sha256:2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91

# GitHub Container Registry에 push (공개 필요)
echo $GITHUB_TOKEN | docker login ghcr.io -u leelang7 --password-stdin
docker buildx build --platform linux/arm64 --push \
  --file container/Dockerfile.routerx \
  --tag ghcr.io/leelang7/routerx:submission .

# 전체 다이제스트 확보
docker buildx imagetools inspect ghcr.io/leelang7/routerx:submission \
  --format '{{.Manifest.Digest}}'
```

push 후 패키지를 **공개(public)** 로 전환해야 한다. ghcr.io 기본값은 비공개다.

## 4. 공식 실행 검증 (push한 이미지로)

```console
PYTHONPATH=src python3 tools/check_runtime.py \
  --image ghcr.io/leelang7/routerx@sha256:<다이제스트> \
  --report build/runtime-check-report.json
```

## 5. 기술 정보 파일 커밋

저장소 루트에 `submission-ossp-skt.json`을 만든다. 여섯 필드만 허용한다.

```json
{
  "schema_version": 1,
  "challenge_id": "ossp-2026-llm-router-challenge",
  "repository_url": "https://github.com/leelang7/ossp-2026-llm-router-challenge",
  "commit_sha": "<2단계에서 얻은 40자리>",
  "image_digest": "ghcr.io/leelang7/routerx@sha256:<64자리>",
  "primary_license": "Apache-2.0"
}
```

```console
python3 tools/validate_technical_submission.py     # 통과해야 한다
git add submission-ossp-skt.json
git commit -m "chore: 기술 제출 정보 추가"
git push origin routerx
git rev-parse HEAD          # ← 이 커밋의 스냅샷 URL을 보고서에 적는다
```

## 6. 결과보고서 제출

- 원본(HWP/DOCX) 1부 + PDF 1부, 본문 5페이지 이내, 맑은고딕 10pt
- 파일명: `2026 오픈소스 개발자대회 결과보고서_접수번호(팀명)`
- 첫 쪽 작성 안내와 회색 안내 문구는 **삭제**
- 프로젝트 등록 URL:
  `https://github.com/leelang7/ossp-2026-llm-router-challenge/tree/<5단계 커밋 SHA>`
- 시연영상: 유튜브 3분 이내, URL 기재
- 붙임1 SBOM, 붙임2 AI 모델 명세 포함 (초안은 `REPORT_DRAFT.md`)
- osscontest.kr에 업로드

## 확인 목록

- [ ] fork가 공개이고 제출 커밋을 권한 없이 열 수 있다
- [ ] 이미지가 `@sha256:` 전체 다이제스트로 참조된다(태그 아님)
- [ ] 이미지가 `linux/arm64`이고 `VOLUME` 선언이 없다
- [ ] 공개 Train+Dev 전체로 세 등급 실행 시간과 출력 형식을 확인했다
- [ ] `submission-ossp-skt.json`이 검증을 통과하고 최종 커밋에 있다
- [ ] 보고서의 등록 URL이 그 JSON을 포함한 커밋을 가리킨다
- [ ] 저장소·이미지에 포함한 파일의 라이선스 근거가 공개되어 있다
      (`container/BASE_IMAGE_ROUTERX.md`, `REUSE.toml`)

## 사용자 결정이 필요한 항목

1. **팀명** — 상장에 그대로 인쇄된다. 특수문자 제외.
2. **시연영상** — 3분 이내 촬영·업로드.
3. **레지스트리** — ghcr.io(GitHub 계정 연동, 무료 공개) 사용 여부.
