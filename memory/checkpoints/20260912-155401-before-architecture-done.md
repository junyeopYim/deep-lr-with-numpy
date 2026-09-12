# Checkpoint — 기본 아키텍처 최종 확인 — 2026-09-12

## The story so far
05–10 여섯 노트북과 그림 helper를 모두 만들었습니다. 코드 109개·그림 30개, notebook 오류·글리프 경고 0개를 확인했습니다.
공통 기초도 새 실행기 옵션으로 재실행하여 147개·31개를 확인했고 pytest 19개가 통과했습니다.
독립 실행에서 CNN 미완성 반환 계약, Transformer mask의 배치/head 축, GNN의 상수 특성 조건을 보완했습니다.
GNN에 실제 고립 노드를 추가하고 그림 노드 잘림을 고쳤습니다. 최신 09·10의 독립 확인과 전체 아키텍처 마지막 실행이 진행 중입니다.
README와 CURRICULUM에 여섯 실제 링크를 연결했습니다. 검증 보고서와 최신 소스 해시·상태 마무리가 남았습니다.

## Decided
D-006의 05–10 전체 제작. 데이터는 본문에서 생성하고 핵심 수학·학습 코드는 본문에 둡니다.

## Waiting on the user
추가 결정 없음.

## Next first action
진행 중인 scripts/verify_notebooks.py --suite architecture 실행 결과와 rehearsal09-final·rehearsal10-final 보고서를 확인하고 results/architecture-validation/architectures.md를 완성합니다.

## Tried
CNN 검사기의 zip에 빈 반환이 통과해 명시적 3개 반환·shape 검사를 추가했습니다.
MHA에 (N,T,T) mask를 주면 N=head일 때 다른 배치가 섞이므로 2D/4D 입력만 허용하고 배치별 mask 검산을 추가했습니다.
일렬 그래프의 y축 자동 범위로 노드가 잘려 좌표 범위에 여백을 명시했습니다.
GNN 선형 readout의 구분 불가능성은 상수 특성 조건이 필요하며, 단순 zero-in-degree와 isolated를 구분했습니다.
