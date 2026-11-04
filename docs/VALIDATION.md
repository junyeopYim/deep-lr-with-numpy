# 2026-11-04 커리큘럼 완성 검증

기존 43개에 남아 있던 **28개 단원**을 추가해 번호 00–63과 b 단위 7개, **총 71개**를 완성했습니다. 아래 28개는 각 본문과 정답을 채운 사본을 **서로 다른 새 커널**에서 위부터 끝까지 실행했습니다. 본문의 의도적인 미작성 연습 안내만 허용했고, 정답 실행은 각 9개·전체 252개를 모두 통과했습니다.

검증 환경은 Python 3.11.5·NumPy 2.4.6·CPU입니다. 실행 예외·미실행 셀·stderr·한글 글리프 경고는 0개입니다. 그림은 PNG 디코딩과 실제 화면 검토를 마쳤습니다. 전체 파라미터 미분·독립 계산·실제 학습·평가는 각 본문에 남겨 두었습니다. 이 확인은 작은 교육 예제의 실행과 계산 검증이며, 원논문 성능의 재현이나 보편적인 알고리즘 우열을 의미하지 않습니다.

공통 검증 도구 테스트는 45개를 통과했습니다. 새 표현·언어·강화학습·종합 suite와 전체 suite의 자동 발견, checkpoint 제외도 포함합니다. 기존 43개 노트북은 이번 작업 시작 시점의 해시와 같음을 확인했습니다. 이번 실행 검증 범위는 새로 추가한 28개 단원입니다.

| 단원 | 코드 셀 | 그림 | 정답 | main 커밋 |
|---|---:|---:|---:|---|
| [31 에너지 기반 모델과 MCMC](../notebooks/04_생성/31_energy_mcmc.ipynb) | 51 | 8 | 9/9 | [4bb1508](https://github.com/junyeopYim/deep-lr-with-numpy/commit/4bb1508483c4d1b472c9ad7f91c6f17e1cf46ae6) |
| [36 Flow matching과 ODE 생성](../notebooks/04_생성/36_flow_matching.ipynb) | 56 | 9 | 9/9 | [cef4abb](https://github.com/junyeopYim/deep-lr-with-numpy/commit/cef4abbae24803e52f900ed4bbd8d622bcb7d4b1) |
| [38 데이터 변형과 불변성](../notebooks/05_표현학습/38_augmentations.ipynb) | 51 | 7 | 9/9 | [0695f83](https://github.com/junyeopYim/deep-lr-with-numpy/commit/0695f83ff5eeb72ebf68019ad6740a9611745e6d) |
| [39 대조학습과 InfoNCE](../notebooks/05_표현학습/39_contrastive_infonce.ipynb) | 57 | 8 | 9/9 | [805e08d](https://github.com/junyeopYim/deep-lr-with-numpy/commit/805e08deae8eff70d6ce0339d61b0a2eaef7440b) |
| [40 교사·학생·stop-gradient](../notebooks/05_표현학습/40_teacher_student.ipynb) | 57 | 7 | 9/9 | [4977108](https://github.com/junyeopYim/deep-lr-with-numpy/commit/49771086107ae49a66f2f203da5869360f72bfbd) |
| [41 분산·공분산으로 표현 제약하기](../notebooks/05_표현학습/41_variance_covariance.ipynb) | 57 | 7 | 9/9 | [210b899](https://github.com/junyeopYim/deep-lr-with-numpy/commit/210b89963825c567b960012b2187f9149b886785) |
| [42 masked reconstruction](../notebooks/05_표현학습/42_masked_reconstruction.ipynb) | 54 | 8 | 9/9 | [c30023a](https://github.com/junyeopYim/deep-lr-with-numpy/commit/c30023af7dee89724a18ba322249a857f9f013d5) |
| [43 JEPA와 잠재 타깃 예측](../notebooks/05_표현학습/43_jepa.ipynb) | 61 | 9 | 9/9 | [3c18504](https://github.com/junyeopYim/deep-lr-with-numpy/commit/3c185042da6784621936740de5e91294d4afa184) |
| [44 토큰화·언어 데이터·자기회귀 목적](../notebooks/06_언어모델/44_tokenization.ipynb) | 50 | 6 | 9/9 | [ce543e8](https://github.com/junyeopYim/deep-lr-with-numpy/commit/ce543e81d5d7eaf44810607141411a47596b8525) |
| [45 작은 decoder GPT 전체 학습](../notebooks/06_언어모델/45_decoder_gpt.ipynb) | 57 | 6 | 9/9 | [4a79b07](https://github.com/junyeopYim/deep-lr-with-numpy/commit/4a79b07e9bc02baaad51b63106c88271b8f9d967) |
| [46 생성 규칙과 KV cache](../notebooks/06_언어모델/46_generation_kv_cache.ipynb) | 63 | 6 | 9/9 | [1dfaf91](https://github.com/junyeopYim/deep-lr-with-numpy/commit/1dfaf91231801655dbedc9161b19efdddd530267) |
| [47 encoder 목적과 BERT식 MLM](../notebooks/06_언어모델/47_masked_language_model.ipynb) | 59 | 6 | 9/9 | [a01b5ea](https://github.com/junyeopYim/deep-lr-with-numpy/commit/a01b5eaa6458c717ccba3fda7bdb1e33d7bd5bb4) |
| [48 SFT·전이 학습·LoRA](../notebooks/06_언어모델/48_sft_lora.ipynb) | 63 | 6 | 9/9 | [924d83c](https://github.com/junyeopYim/deep-lr-with-numpy/commit/924d83cb68ff4d3715ce3fa4baca57f09fa8e85e) |
| [49 선호 쌍에서 DPO 목적 만들기](../notebooks/06_언어모델/49_dpo.ipynb) | 66 | 6 | 9/9 | [b38144f](https://github.com/junyeopYim/deep-lr-with-numpy/commit/b38144faeb9dc57d1779fe82103efd015df0b7c4) |
| [50 이미지·텍스트 정렬과 조건부 생성](../notebooks/06_언어모델/50_multimodal.ipynb) | 69 | 7 | 9/9 | [4948552](https://github.com/junyeopYim/deep-lr-with-numpy/commit/4948552e3df4178c8789d83c4c9d3137af57d494) |
| [51 Bandit·기대 보상·탐색](../notebooks/07_강화학습/51_bandits.ipynb) | 46 | 7 | 9/9 | [250c8a3](https://github.com/junyeopYim/deep-lr-with-numpy/commit/250c8a36c42a72627e0dd0c70b6b693cd5cc6c89) |
| [52 MDP·return·Bellman·동적 계획](../notebooks/07_강화학습/52_mdp_bellman.ipynb) | 49 | 8 | 9/9 | [b4e9d87](https://github.com/junyeopYim/deep-lr-with-numpy/commit/b4e9d87dd5ac2a1e3e40aeaa0e946a334c254c11) |
| [53 Monte Carlo·TD·다단계 타깃](../notebooks/07_강화학습/53_mc_td.ipynb) | 51 | 7 | 9/9 | [ead9de1](https://github.com/junyeopYim/deep-lr-with-numpy/commit/ead9de1d5a49bf7393c088b8cbceb0f78d08a358) |
| [54 SARSA·Expected SARSA·Q-learning](../notebooks/07_강화학습/54_sarsa_q_learning.ipynb) | 48 | 7 | 9/9 | [9774b79](https://github.com/junyeopYim/deep-lr-with-numpy/commit/9774b7935372d6eff7ed764601c9c51dc2c612e9) |
| [55 DQN·replay·target network](../notebooks/07_강화학습/55_dqn.ipynb) | 54 | 7 | 9/9 | [aaeadee](https://github.com/junyeopYim/deep-lr-with-numpy/commit/aaeadee4858177989bdcf58405fa4a73aaab5a0e) |
| [56 REINFORCE와 정책 기울기 유도](../notebooks/07_강화학습/56_reinforce.ipynb) | 50 | 8 | 9/9 | [7051ea4](https://github.com/junyeopYim/deep-lr-with-numpy/commit/7051ea433724112b6a4164c85912416cd6845688) |
| [57 Actor–Critic·advantage·GAE](../notebooks/07_강화학습/57_actor_critic_gae.ipynb) | 49 | 8 | 9/9 | [1895d8a](https://github.com/junyeopYim/deep-lr-with-numpy/commit/1895d8a31b2e92ddc43ca683e1bc8acd817a2d3d) |
| [58 PPO의 ratio·clip·업데이트](../notebooks/07_강화학습/58_ppo.ipynb) | 49 | 8 | 9/9 | [672da7d](https://github.com/junyeopYim/deep-lr-with-numpy/commit/672da7d52e4d599649db7037f59ef17896a9dd33) |
| [59 연속 행동·DDPG/TD3·SAC](../notebooks/07_강화학습/59_continuous_control.ipynb) | 62 | 8 | 9/9 | [e03d0c3](https://github.com/junyeopYim/deep-lr-with-numpy/commit/e03d0c3ec96661bb3162a5525948ec21561a58df) |
| [60 모델 기반 RL과 계획](../notebooks/07_강화학습/60_model_based_rl.ipynb) | 56 | 9 | 9/9 | [2abf97d](https://github.com/junyeopYim/deep-lr-with-numpy/commit/2abf97d766710e782cdce8bfb4fb3aadabf00947) |
| [61 오프라인 RL과 정책 평가](../notebooks/07_강화학습/61_offline_rl.ipynb) | 63 | 9 | 9/9 | [5bed46f](https://github.com/junyeopYim/deep-lr-with-numpy/commit/5bed46fbab76e8e9e2c18d36fa005aa6ab1ff5fe) |
| [62 선호 보상과 RLHF 연결](../notebooks/07_강화학습/62_rlhf.ipynb) | 73 | 7 | 9/9 | [b364f35](https://github.com/junyeopYim/deep-lr-with-numpy/commit/b364f35d70dcd7d71ade78197430a430eca29c52) |
| [63 논문에서 수식·구현·실험으로](../notebooks/08_종합/63_paper_reproduction.ipynb) | 49 | 8 | 9/9 | [9f0bee6](https://github.com/junyeopYim/deep-lr-with-numpy/commit/9f0bee6030759ebb098fdf193bd3fb9bd85376ad) |

총 그림은 **207장**입니다. 단원마다 노트북·그림 보조 코드·커리큘럼/작성/그림 문서를 별도 커밋하고 main에 푸시했습니다. 커밋에 들어간 노트북·보조 코드의 SHA-256과 최종 파일을 대조했습니다.

로컬 상세 로그와 그림은 `results/full-curriculum-2026-11-04/`에 있습니다. `NN-original.log`, `NN-answers.log`, `NN-comments.log`, `NN-figures/`, `NN-validation.json`을 함께 확인할 수 있습니다. 이 폴더는 Git에서 제외합니다.

재검증은 저장소 루트의 같은 Python 환경에서 실행합니다.

```bash
python scripts/verify_notebooks.py --suite all
python scripts/verify_exercises.py notebooks/*/*.ipynb
python scripts/check_code_comments.py notebooks/*/*.ipynb
python -m pytest
```

본문 검증은 실행 결과를 노트북에 저장합니다. 정답 검증은 메모리 사본에만 정답을 채우고 원본 연습은 비워 둡니다. 전체 검증에는 기존 MNIST 학습 등 오래 걸리는 단원도 포함됩니다. 한 단원만 확인하려면 해당 노트북 경로를 검증 명령에 전달하세요.
