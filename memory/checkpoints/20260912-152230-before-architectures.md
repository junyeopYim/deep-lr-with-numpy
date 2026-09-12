# Checkpoint — 공통 기초 00–04 완료 — 2026-09-12

## The story so far
D-005의 공통 기초 다섯 노트북을 모두 제공했습니다.
새 02 확률·손실, 03 퍼셉트론·MLP, 04 optimizer를 작성하고 기존 00·01과 연결했습니다.
최종 147개 코드 셀·31개 그림 실행·확인, 오류·한글 글리프 경고 0개, pytest 19개 통과.
02·03·04는 독립 실행·직접 변형·오답 검출도 통과했습니다.
근거: results/foundation-validation/common-foundations.md와 execution.json.

## Decided
D-004: 수학도 코드와 함께 확인, 작은 완성 예제 뒤 직접 변형.
D-005: 공통 기초 00–04를 각각 실행 가능한 노트북으로 제작.

## Waiting on the user
제작 완료에 필요한 추가 결정 없음.
실제 공부에서 막힌 위치가 생기면 그 반응으로 설명 깊이와 문제 난도를 조정합니다.

## Next first action
후속 제작 요청이 오면 docs/CURRICULUM.md의 05 CNN을 지역 연결·가중치 공유·패치 연산·backward 검산부터 구현합니다.

## Tried
균일 logits만으로 CE를 검사하면 logit을 무시한 풀이가 통과하여 비균일 손계산을 추가했습니다.
MLP 검사기가 원본 backward에 고정돼 backward_fn 인자로 직접 쓴 함수를 전달하도록 고쳤습니다.
Adam의 같은 beta·epsilon=0 사례만으로는 계수·분모 오류를 놓쳐 서로 다른 beta와 작은 기울기 사례를 추가했습니다.
로그·마스크·초기화 분산의 독립성 전개를 본문 안에서 설명하고 검산했습니다.
