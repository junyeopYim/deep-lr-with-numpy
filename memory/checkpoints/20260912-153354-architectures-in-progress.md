# Checkpoint — 기본 아키텍처 05–10 제작 시작 — 2026-09-12

## The story so far
공통 기초 00–04는 147개 코드 셀·31개 그림·pytest 19개 검증 후 완료했습니다.
사용자가 기본 아키텍처 전체 제작을 요청하여 05–10 여섯 노트북을 작성합니다.
작성 소스는 /tmp/dlfs-architectures/, 최종 파일은 notebooks/01_아키텍처/에 둡니다.
검증 기록은 results/architecture-validation/로 기존 공통 기초 기록과 구분합니다.

## Decided
D-006: CNN, RNN, LSTM·GRU, Attention, Transformer, GNN의 여섯 노트북 제작.
D-004: 필요한 수학과 배열 연결, 작은 완성 예제 다음 직접 변형.

## Waiting on the user
제작에 필요한 추가 결정 없음.

## Next first action
/tmp/dlfs-architectures/05.cells에 CNN의 패치·가중치 공유·역전파와 작은 숫자 이미지 분류 실험을 작성합니다.

## Tried
기존 노트북 실행 스크립트는 모든 결과를 foundation-validation에 쓰므로 새 검증 경로 옵션을 먼저 추가합니다.
변형 검사기는 전달받은 실제 풀이를 비균일 입력·수치미분·의도적 오답으로 검사해야 합니다.
