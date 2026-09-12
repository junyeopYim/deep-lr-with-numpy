# Checkpoint — 공통 기초 00–04 최종 점검 중 — 2026-09-12

## The story so far
02 확률·손실, 03 퍼셉트론·MLP, 04 optimizer의 각 노트북과 그림 보조 파일을 작성했습니다.
02·03·04 모두 1차 실행 통과. 02 로그·마스크 설명과 CE 검사기, 03 후보 backward 연결과 연습 제목을 보강했습니다.
README·커리큘럼에 모든 링크를 연결했고 지금 다섯 노트북 전체 실행과 최종 독립 점검 중입니다.

## Decided
D-004: 작은 완성 예제 → 수식·코드·검산 → 직접 변형.
D-005: 공통 기초 00–04 각각 노트북으로 제작.

## Waiting on the user
필요한 추가 결정 없음.

## Next first action
/tmp/dlfs-common-foundations/all-execution.log와 rehearsal04/report.md, rehearsal02-03-final/report.md를 확인하고 남은 오류를 고칩니다.

## Tried
균일 logits만 검사하면 잘못된 CE도 통과해 비균일 손계산을 추가했습니다.
MLP 검사기가 원본 backward에 고정되어 backward_fn 인자로 학습자의 함수를 전달하도록 수정했습니다.
