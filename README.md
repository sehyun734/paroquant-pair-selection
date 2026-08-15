# ParoQuant Shuffle Ablation

[github](https://github.com/z-lab/paroquant) · [arxiv](https://arxiv.org/abs/2511.10645)

## 동기

논문은 Motivation 섹션 세번째 문단에서 magnitude 차이가 큰 pair, 즉 outlier와 normal channel을 짝지어 rot을 거는 것이 효과적이라 주장함

> Intuitively, rotations between an outlier channel and a normal channel would be more effective at reducing outliers than rotations between two normal channels.

> ... optimizing only the top 10% channel pairs with largest magnitude difference is almost as effective as optimizing all pairs (Figure 2)

그러나 논문 A.1 알고리즘에서도, [실제 구현](https://github.com/z-lab/paroquant/blob/legacy/paroquant/optimize.py)에서도 크기는 보지 않고 후보를 통째로 shuffle함...

> P_shuffled ← Shuffle(P)

> For each rotation, we randomly select available pairs while ensuring the rotation remains independent.

```py
def get_random_rotation_pairs(sensitivity_input, ...):
    sorted_pairs = []
        ...
        rand.shuffle(sorted_pairs[group_idx])
```

## 사전 조사

경위는 논문에도 GitHub 이슈에도 없고 [OpenReview 저자 답변](https://openreview.net/forum?id=1USeVjsKau)에서 확인

제출본은 random이 아니라 greedy. 리뷰어 hpqW가 그마저도 전역 최적이 아닐 수 있다고 지적

> The greedy pair selection strategy outlined in Algorithm A1, while effective and intuitive, may not be globally optimal.

저자는 rot이 독립인 한 선별 전략에 robust하다고 답변. 모든 outlier가 적어도 한 번은 normal channel과 짝지어짐을 확률로 보이고, 그 김에 구현도 단순함을 위해 random으로 바꿨다고 덧붙임

> ... independent rotations is highly robust to the pair selection strategy. In practice, greedy selection or even random pairing is nearly optimal for our primary goal

> Although the original design used a greedy strategy, our updated implementation adopts random selection for simplicity, and our experiments on perplexity and downstream reasoning tasks confirm that this change does not affect end-to-end performance.

전환은 ICLR 심사 중 rebuttal에서 일어남. random은 처음부터의 설계가 아니라 사후에 정당화된 단순화. 알고리즘만 갈아끼우고 motivation은 방치하니 둘이 어긋난 것...

## 설계

저자 답변이 이미 있으니 방향을 틀어 그 답변을 검증. 선별 전략이 무관하다면 어떤 arm으로 짝을 골라도 loss가 비슷해야 함

모델 전체를 양자화해 ppl을 재는 것이 정공법이지만 느림. 저자도 rebuttal Q4에서 linear 하나의 출력 loss로 대신했으니 그대로 따름. 변환 없이 양자화한 RTN(round-to-nearest)을 100%로 두고 상대값으로 비교

loss와 최적화 설정은 Appendix A.2를 따름. quantization-induced output error, AdamW, lr 0.01, n_steps=200, calibration은 Pile. n_seqs만 시간을 아끼려고 128에서 16으로 축소

| linear | scaling | +8 IR |
| --- | --- | --- |
| Qwen3-0.6B q_proj | 78 | 53 |
| Qwen3-0.6B gate_proj | 75 | 55 |

rebuttal Q4의 이 표로 실험 환경을 우선 재현. scaling은 alpha만 쓰고 rot은 안 붙인 것, +8 IR은 그 위에 독립 rot 8개를 붙인 것. 환경이 맞으면 pair 고르는 기준만 아래 여섯으로 갈아끼움

| arm | 기준 |
| --- | --- |
| rand | 무작위 셔플 (baseline, 현행 구현) |
| butterfly | 결정론적 패턴. 무작위도 정렬도 아니라 rand의 우위가 무작위성에서 오는지 가르는 통제군 |
| mag_diff | 크기 차이 큰 순서 (논문 motivation) |
| mag_sim | 크기 차이 작은 순서 (논문 동기와 정반대) |
| corr | 상관관계 높은 순서 |
| a_mag | 입력 크기를 반영한 중요도 차이 큰 순서 (AWQ식 activation-aware) |

butterfly는 채널 번호를 stride로 XOR해 짝지음. grp_size 128이면 stride가 7개뿐이라 n_rots=8의 여덟 번째는 첫 번째와 같은 pair를 다시 씀. theta는 rot마다 따로 붙으니 rot 수는 여덟이 맞고, pair가 겹치는 것은 뒤에서 볼 연결 구조에만 영향

저자 주장이 맞다면 여섯 모두 loss가 비슷해야 함. 이때 rand만 무작위성을 타므로 seed 5개로 돌려 최소~최대 혹은 평균으로 보고

다만 이미 쓴 pair를 건너뛰는 제약 탓에 일부 rot은 pair를 다 채우지 못함. 그러면 loss가 나쁜 이유가 arm이 아니라 rot을 덜 한 탓일 수도 있음. 논문은 무시할 수준이라고 넘어가긴 했음

> This constraint may result in an insufficient number of pairs for some rotations, but the impact is negligible in practice.

그래도 일단 dummy도 함께 출력해 상황에 맞게 판단

## 실험

### 환경 재현

Qwen3-0.6B layer=0 q_proj·gate_proj [scaling norm](runs/loss_260807-030534.log) · [scaling mse](runs/loss_260807-031045.log) · [rand](runs/loss_260807-035319.log). n_steps=200. rand는 n_rots=8, seed 최소~최대

| linear | scaling (저자) | norm | mse | +8 IR (저자) | rand |
| --- | --- | --- | --- | --- | --- |
| q_proj | 78 | 87.1 | 76.0 | 53 | 52.6~53.7 |
| gate_proj | 75 | 85.7 | 73.7 | 55 | 54.2~55.1 |

- **저자 환경은 mse**. A.2 표기대로 norm으로 재면 저자 값과 9 이상 벌어지지만 mse면 2 이내. rot을 붙인 +8 IR도 저자 값이 rand의 seed 범위 안

환경이 맞춰졌으니 이후 측정은 전부 mse

### arm별 loss

[q_proj layer=0/10/20 n_rots=8](runs/loss_260808-154356.log). rand seed 평균 대비 편차, model(4) 평균. 괄호는 dummy 최소~최대, rot 없는 scaling은 -

| layer | rand (절대) | scaling | butterfly | mag_diff | mag_sim | corr | a_mag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 34.7 (2~18) | +33.8 (-) | +0.9 (0) | +3.2 (0) | +5.4 (27~137) | +2.4 (3~33) | +3.3 (0~129) |
| 10 | 41.8 (2~18) | +16.5 (-) | +0.1 (0) | +2.8 (0) | +2.4 (32~129) | +0.6 (5~15) | +2.6 (0) |
| 20 | 46.7 (2~18) | +16.4 (-) | +0.2 (0) | +2.1 (0) | +2.3 (29~141) | +0.9 (2~14) | +2.5 (0) |

- **정렬하면 손해**. 정렬하는 넷이 전부 rand보다 뒤처짐. 논문 motivation인 mag_diff도 예외 아님
- **무작위성은 원인 아님**. 무작위도 정렬도 하지 않는 butterfly가 layer 10/20에서 +0.1~+0.2로 rand와 동률
- **dummy 탓도 아님**. rand는 dummy가 2~18로 mag_diff(0)보다 많은데 loss는 더 낮음
- **layer 0만 편차가 큼**. 다섯 arm이 +0.9~+5.4로 벌어짐. layer 10/20은 +0.1~+2.8이고, a_mag의 dummy가 129까지 가는 것도 layer 0뿐

layer 0은 나머지 두 layer와 양상이 달라 이후 측정은 layer 10/20만 씀

무작위인 rand와 결정론적인 butterfly가 나란히 이겼으니, 원인은 pair 하나하나의 품질보다 pair들이 이루는 구조 쪽일 수 있음. rot 1이 A·B를, rot 2가 B·C를 짝지으면 A와 C도 간접적으로 이어짐. 반면 정렬은 pair 순서가 고정이라 rot을 겹쳐도 비슷한 채널끼리만 묶임. 연결성을 확인해 봄

### arm별 연결성

| 지표 | 뜻 | 
| --- | --- |
| blocks | 그룹당 덩어리 수. 1이면 grp_size=128채널이 전부 하나로 이어짐 |
| lam2 | Laplacian의 두 번째로 작은 고유값. 0이면 끊김, 클수록 촘촘하게 이어짐 |

[q_proj layer=10/20 n_rots=1/2/4/8](runs/graph_260808-211801.log). 값은 blocks / lam2. model(4) × layer(2) 8셀 최소~최대. dummy는 rot하지 않으므로 제외

| n_rots | rand | butterfly | mag_diff | corr | a_mag |
| --- | --- | --- | --- | --- | --- |
| 1 | 64 / 0.00 | 64 / 0.00 | 64 / 0.00 | 64 / 0.00 | 64 / 0.00 |
| 2 | 2.5~2.9 / 0.00 | 32 / 0.00 | 32 / 0.00 | 2.3~3.6 / 0.00 | 32 / 0.00 |
| 4 | 1 / 0.63~0.66 | 8 / 0.00 | 16 / 0.00 | 1 / 0.52~0.60 | 16 / 0.00 |
| 8 | 1 / 3.00~3.03 | 1 / 2.00 | 8 / 0.00 | 1 / 2.36~2.87 | 8 / 0.00 |

- **정렬하면 그래프가 갈라짐**. rand·corr는 rot 4, butterfly는 rot 8에 하나로 이어짐. mag_diff·a_mag는 끝까지 갈라진 채로 남음
- **n_rots=8에서 loss와 순서가 같음**. [arm별 loss](#arm별-loss) layer 10/20에서 이어진 셋(butterfly +0.1~+0.2, corr +0.6~+0.9)이 끊긴 둘(mag_diff +2.1~+2.8, a_mag +2.5~+2.6)보다 낮음

rand가 잘 이어지고 mag_diff·a_mag가 끊긴 것은 예상대로. 다만 정렬하는 corr도 rand만큼 이어져 예상과 다름. 어느 쪽이든 연결성 차이가 없는 n_rots=1에서 정렬이 앞서면 확실해짐

### n_rots별 loss

[q_proj layer=10/20 n_rots=1/2/4/8](runs/loss_260809-003523.log). rand seed 평균 대비 편차, model(4) × layer(2) 8셀 평균. mag_sim은 dummy 탓에 제외

| n_rots | rand (절대) | butterfly | mag_diff | corr | a_mag |
| --- | --- | --- | --- | --- | --- |
| 1 | 55.7 | −0.0 | +0.4 | +2.2 | −0.8 |
| 2 | 51.7 | +0.1 | +1.5 | +2.6 | +0.7 |
| 4 | 47.6 | +0.3 | +2.4 | +1.8 | +2.3 |
| 8 | 44.2 | +0.2 | +2.5 | +0.8 | +2.6 |

- **연결성은 원인 아님**. n_rots=1은 다섯 arm의 blocks·lam2가 전부 같은데도 −0.8~+2.2로 벌어짐. n_rots=2에서도 연결성이 rand와 같은 corr가 오히려 +2.6으로 최악
- **pair 품질 이득은 있음**. a_mag가 n_rots=1에서 −0.8로 rand를 이김
- **다만 누적되지 않음**. rot을 겹칠수록 a_mag는 −0.8 → +2.6, mag_diff는 +0.4 → +2.5로 거꾸로 감. rot을 더 붙일 때 늘어나는 이득을 rand만큼 못 가져감
- **corr만 방향 반대**. +2.2 → +0.8로 좁혀짐
- butterfly는 전 구간 rand와 동률

연결성 가설은 기각. 원인이 안 좁혀지니 일단 rng를 한번 봐봄

### n_rots별 rng

| 지표 | 뜻 |
| --- | --- |
| rng | 그룹 안 최대−최소의 제곱 평균, RTN 대비 % |

[q_proj layer=10/20 n_rots=1/2/4/8](runs/loss_260809-163126.log). 값은 loss / rng. rand seed 평균 대비 편차, model(4) × layer(2) 8셀 평균. mag_sim은 dummy 탓에 제외

| n_rots | rand (절대) | butterfly | mag_diff | corr | a_mag |
| --- | --- | --- | --- | --- | --- |
| 1 | 55.7 / 64.8 | −0.0 / −0.3 | +0.4 / +0.7 | +2.2 / +2.4 | −0.8 / −0.8 |
| 2 | 51.7 / 58.5 | +0.1 / −0.1 | +1.5 / +2.0 | +2.6 / +3.1 | +0.7 / +1.8 |
| 4 | 47.6 / 52.1 | +0.3 / −0.0 | +2.4 / +3.6 | +1.8 / +2.2 | +2.3 / +4.1 |
| 8 | 44.2 / 47.0 | +0.2 / +0.4 | +2.5 / +3.7 | +0.8 / +0.9 | +2.6 / +4.4 |

- **loss가 rng를 그대로 따라감**. [n_rots별 loss](#n_rots별-loss)에서 a_mag가 −0.8 → +2.6으로 간 것은 rng가 −0.8 → +4.4로 간 것. 거꾸로 좁혀진 corr도 rng가 +2.4 → +0.9로 같이 좁혀짐
- **정렬은 rng를 못 줄임**. 절대값으로도 rand는 64.8 → 47.0으로 17.8을 줄이는데 a_mag는 12.6에 그침

rng는 다를 줄 알았는데 순위도 방향도 앞 절 그대로. 생각해보면 loss·rng 둘 다 같은 가중치를 요약한 값이라 지표를 아무리 늘려도 같은 해석만 나옴. 지표를 늘리는 쪽은 접고, 일단 정렬의 손해가 최적화가 덜 된 탓인지 확인해 봄

### n_steps별 loss

q_proj layer=10 n_rots=8 [Llama-3.1-8B](runs/loss_260810-172628.log) · [Mistral-7B-v0.3](runs/loss_260810-181059.log). n_steps=200은 [n_rots별 rng](#n_rots별-rng)로부터. 값은 loss / rng. rand seed 평균 대비 편차, model별. 괄호는 dummy 최소~최대. mag_sim은 dummy 탓에 제외

| model | n_steps | rand (절대) | butterfly | mag_diff | corr | a_mag |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | 200 | 52.5 / 35.4 (8~18) | +0.3 / +0.2 (0) | +3.9 / +5.5 (0) | +0.4 / +0.5 (15) | +3.9 / +6.3 (0) |
| | 800 | 49.6 / 32.7 (8~18) | +0.1 / −0.0 (0) | +4.6 / +7.4 (0) | +0.2 / +0.6 (15) | +4.6 / +8.2 (0) |
| Mistral-7B-v0.3 | 200 | 38.7 / 38.7 (8~18) | +0.3 / +0.5 (0) | +2.8 / +4.0 (0) | +0.9 / +0.8 (8) | +2.8 / +4.6 (0) |
| | 800 | 35.6 / 34.6 (8~18) | −0.1 / +0.1 (0) | +3.8 / +6.7 (0) | +0.7 / +1.3 (8) | +3.9 / +7.2 (0) |

- **최적화 탓 아님**. n_steps=800에서 mag_diff·a_mag의 격차가 오히려 벌어짐. Llama는 +3.9 → +4.6, Mistral은 +2.8 → +3.8
- **arm이 두 집단으로 갈림**. 수렴시킬수록 butterfly·corr는 −0.1~+0.7로 rand에 붙고, mag_diff·a_mag는 +3.8 밖으로 떨어짐

정렬의 손해는 pair 선별 자체 문제가 맞나 봄

남는 건 n_rots가 늘어날수록 정렬이 불리해진다는 것. 그런데 생각해보니 rot 2부터도 처음 매긴 점수가 그대로 남아 있는지는 확인한 적이 없음. 이미 쓴 pair는 건너뛰고, rot가 `w`를 바꾸니 같은 pair라도 점수가 달라짐. 확인해 봄

### rot별 pair 점수

| 지표 | 뜻 |
| --- | --- |
| orig | 원본 `w`에서 처음 pair 고를 때 채널 쌍 점수 |
| hit | rot 전(alpha만 걸린) 대비 실제 rot 시점의 점수 비 |

[Llama-3.1-8B q_proj layer=10 n_rots=8](runs/pairs_260811-173134.log). rng의 0은 alpha만 걸고 rot는 안 한 시점. loss는 같은 조건의 [n_rots별 rng](#n_rots별-rng) 로그에서. rand는 seed 최소~최대, dummy는 rot별 합. 점수 없는 rand·butterfly는 orig·hit 제외

| arm | dummy | orig (1 → 8) | hit (8) | rng (0 → 8) | loss |
| --- | --- | --- | --- | --- | --- |
| rand | 8~18 | - | - | 54.4~54.8 → 35.2~35.6 | 52.5~52.6 |
| butterfly | 0 | - | - | 54.7 → 35.6 | 52.8 |
| mag_diff | 0 | 8.33e-04 → 8.33e-04 | 54% | 56.3 → 40.9 | 56.4 |
| mag_sim | 129 | −7.09e-05 → −2.05e-04 | 68% | 55.7 → 40.3 | 56.4 |
| corr | 15 | 8.43e-02 → 5.99e-02 | 69% | 54.7 → 35.9 | 52.9 |
| a_mag | 0 | 1.65e-03 → 1.65e-03 | 71% | 56.9 → 41.7 | 56.4 |

- **pair가 부족하진 않음**. mag_diff·a_mag는 orig가 8 rot 내내 불변, dummy도 0. 원본 기준으로는 rot 8이 고른 pair도 rot 1만큼 좋음
- **다만 쓸 때쯤이면 점수가 안 남음**. hit이 54~71%. mag_diff의 rot 8은 rot 전 9.34e-04에서 앞선 일곱을 겪고 5.02e-04. rot이 하는 일이 크기 차이를 줄이는 것이라 겹칠수록 점수 자체가 깎임
- **그런데 이것도 loss와 안 묶임**. 가장 잘 버틴 a_mag(71%)가 loss 56.4로 최악, 비슷하게 버틴 corr(69%)는 52.9로 최고. 절반만 남은 mag_diff(54%)도 56.4로 a_mag와 동률
- **rot 전 alpha에서 이미 격차**. alpha만 걸고 rot는 안 한 시점에서 mag_diff·a_mag가 56.3~56.9로 rand·butterfly·corr(54.4~54.8)보다 불리

신기하게도 손해의 시작점이 rot이 아니라 그 앞의 alpha. 그래서 한번 alpha를 묶어놓고 theta만 최적화하면 어떻게 되는지 확인

### alpha별 loss

| alpha | 뜻 |
| --- | --- |
| joint | alpha·theta 함께 학습. 현행 구현이고 앞 절까지 전부 이것 |
| off | alpha를 1로 고정하고 theta만 학습 |
| pre | rot 없이 alpha를 먼저 학습해 고정한 뒤 theta만 학습 |

[q_proj layer=10/20 n_rots=8](runs/loss_260811-225323.log). 값은 loss / rng. rand seed 평균 대비 편차, model(4) × layer(2) 8셀 평균. mag_sim은 dummy 탓에 제외

| arm | joint | off | pre |
| --- | --- | --- | --- |
| rand (절대) | 44.2 / 47.0 | 81.2 / 86.7 | 47.6 / 58.2 |
| butterfly | +0.2 / +0.4 | +0.3 / +0.0 | +0.1 / +0.2 |
| mag_diff | +2.5 / +3.7 | +0.9 / +0.6 | +1.3 / +1.2 |
| corr | +0.8 / +0.9 | +1.1 / +0.5 | +0.4 / +0.4 |
| a_mag | +2.6 / +4.4 | +0.6 / +0.5 | +1.5 / +1.1 |

- **alpha와 함께 학습하면 격차가 커짐**. joint에서 +2.5~+2.6이던 mag_diff·a_mag가 off에선 +0.6~+0.9, pre에선 +1.3~+1.5로 내려감
- **다만 pair 선별 자체도 손해**. alpha를 통째로 걷어낸 off에서도 정렬 arm이 +0.6~+1.1로 뒤처짐
- **alpha를 바꿔도 두 집단 그대로**. butterfly는 세 alpha 전부 +0.1~+0.3, corr도 +0.4~+1.1. rand에 붙어 있던 arm은 계속 붙어 있음

그럼 정렬의 손해는 pair 선별이 직접 만든 몫(off +0.6~+0.9)과 alpha와 얽히며 커진 몫(joint +2.5~+2.6) 둘로 볼 수 있음

하지만 바뀐 건 없음. 어느 쪽이든 원인은 결국 pair 선별. 기각되는 것들과 현상만 늘어남...

그러다 arm 자체를 의심한 적이 없음을 깨달음. 지금 arm들은 전부 대리지표. "크기 차이가 크면 rng가 잘 줄겠지" 같은 직감으로 고른 것일 뿐, 정말 rng를 가장 잘 줄이는 pair인지는 잰 적이 없음. rng를 직접 보고 고르는 arm을 붙여 봄

### oracle loss

| arm | 기준 |
| --- | --- |
| oracle | pair 하나를 각도 16개로 돌려보고, rng를 가장 많이 줄인 각도의 감소량이 점수 |

[q_proj layer=10/20 n_rots=1/2/4/8](runs/loss_260812-181512.log). rand는 [n_rots별 rng](#n_rots별-rng)로부터. 값은 loss / rng. rand seed 평균 대비 편차, model(4) × layer(2) 8셀 평균. 괄호는 dummy

| n_rots | rand (절대) | oracle |
| --- | --- | --- |
| 1 | 55.7 / 64.8 (0) | −1.5 / −1.8 (0) |
| 2 | 51.7 / 58.5 (0~1) | −0.7 / −0.7 (0~4) |
| 4 | 47.6 / 52.1 (0~6) | −0.2 / −0.1 (2~16) |
| 8 | 44.2 / 47.0 (2~18) | −0.0 / +0.2 (11~62) |

- **pair 품질 이득은 사실**. oracle이 전 구간 rand 이상. n_rots=1에서 −1.5로, [n_rots별 loss](#n_rots별-loss)에서 유일하게 rand를 이겼던 a_mag(−0.8)보다 두 배 가까이 앞섬
- **허나 앞선 arm들처럼 n_rots 늘수록 이득이 줄어듦**. −1.5 → −0.0으로, rot을 더 붙일수록 rand가 따라붙어 n_rots=8에서 차이 없음
- dummy가 62까지 가는데도 앞섬. dummy가 원인이 아니라는 근거 하나 더
- **주의할 건 완전한 최적해는 아님**. 각도를 16개 그리드로만 훑고 pair도 하나씩 따로 재므로 근사

"정렬하면 손해"가 "대리지표로 정렬하면 손해"로 좁혀짐. 정렬 자체가 아니라 직감으로 고른 대리지표가 문제였음

남는 건 여기서도 n_rots가 늘수록 이득이 줄어드는 것. 아마 oracle도 점수를 원본 `w`로 처음 한 번만 매기니, 최적인 시점은 rot 1뿐이고 그 뒤로는 rot가 `w`를 바꾸면서 점수가 달라지는 것일 거임. 암튼 그래서 [rot별 pair 점수](#rot별-pair-점수)와 같이 `pairs.py`로 확인

### oracle pair 점수

[Llama-3.1-8B q_proj layer=10 n_rots=8](runs/pairs_260812-185005.log). [rot별 pair 점수](#rot별-pair-점수)와 같은 조건·같은 열

| arm | dummy | orig (1 → 8) | hit (8) | rng (0 → 8) | loss |
| --- | --- | --- | --- | --- | --- |
| oracle | 44 | 1.91e-05 → 1.70e-05 | 14% | 54.7 → 36.3 | 52.8 |

- **pair가 부족하진 않음**. orig가 1.91e-05에서 1.70e-05로 거의 그대로
- **그런데 hit이 무너짐**. rot 8에서 14%로, 다른 arm(54~71%)의 4분의 1
- **정렬해도 alpha는 안 밀림**. rot 전 시작점이 54.7로 rand·butterfly·corr(54.4~54.8)와 같음. 밀린 것은 크기로 정렬하는 둘뿐

예상대로 oracle이 rot를 거듭할수록 점수가 망가짐. 특히나 최적으로 고른 탓에 hit이 다른 arm보다도 더 빨리 무너지고 이득도 같이 사라짐. 그렇다면 매 rot마다 pair를 다시 고르면 이득을 끝까지 가져갈 수 있지 않을까? 한 번 구현해 봄. 

### sched별 loss

그런데 rot마다 최적 pair를 고르려면 앞선 rot이 반영된 `w`로 점수를 재야 함. 곧 앞선 theta가 확정돼 있어야 한다는 뜻. 그래서 theta도 순차로 학습해야 하는데, 이러면 재채점·순차화라는 두 축이 걸림. 갈라서 2×2로 봄. alpha도 같은 이유로 확정돼 있어야 하므로 pre로 둠

| sched | 뜻 |
| --- | --- |
| all | 처음 한 번만 채점, theta는 전체 rot 한꺼번에 학습. 현행 구현이고 앞 절까지 전부 이것 |
| seq | 처음 한 번만 채점, theta는 앞선 rot 얼려가며 하나씩 학습 |
| seq_rescore | rot마다 재채점, theta도 앞선 rot 얼려가며 하나씩 학습 |
| rescore | rot마다 재채점, theta는 전체 rot 한꺼번에 학습 |

이때 rescore는 재채점과 한꺼번에 학습을 동시에 하는데, 이는 seq_rescore를 우선 한 번 돌려 pair 목록만 가져오고 theta는 처음부터 다시 학습하는 방식임

q_proj layer=10/20 n_rots=1/2/4/8 alpha=pre [Qwen3-0.6B](runs/loss_260815-021023.log) · [Llama-3.1-8B](runs/loss_260815-093713.log) · [Mistral-7B-v0.3](runs/loss_260815-035553.log) · [gemma-2-9b](runs/loss_260815-191643.log). 값은 loss / rng. rand seed 평균 대비 편차, model(4) × layer(2) 8셀 평균. sched마다 최적화가 다르므로 각자의 rand를 기준선으로 씀

| sched | n_rots | rand (절대) | mag_diff | oracle |
| --- | --- | --- | --- | --- |
| all | 1 | 56.8 / 67.8 | +0.0 / +0.0 | −1.5 / −2.0 |
| | 2 | 53.5 / 64.6 | +1.0 / +0.7 | −0.9 / −1.6 |
| | 4 | 50.3 / 61.3 | +1.5 / +1.2 | −0.7 / −1.2 |
| | 8 | 47.6 / 58.2 | +1.3 / +1.2 | −0.5 / −0.8 |
| seq | 1 | 56.8 / 67.8 | +0.0 / +0.0 | −1.5 / −2.0 |
| | 2 | 54.3 / 65.4 | +0.7 / +0.6 | −1.0 / −1.6 |
| | 4 | 51.9 / 63.1 | +1.2 / +1.0 | −0.7 / −1.2 |
| | 8 | 50.0 / 61.3 | +1.2 / +1.0 | −0.3 / −0.8 |
| seq_rescore | 1 | 56.8 / 67.8 | +0.0 / +0.0 | −1.5 / −2.0 |
| | 2 | 54.3 / 65.4 | −0.6 / −0.4 | −2.6 / −3.1 |
| | 4 | 51.9 / 63.1 | −0.6 / −0.6 | −3.4 / −4.3 |
| | 8 | 50.0 / 61.3 | −0.4 / −0.3 | −3.9 / −5.0 |
| rescore | 1 | 56.8 / 67.8 | +0.0 / +0.0 | −1.5 / −2.0 |
| | 2 | 53.5 / 64.6 | −0.4 / −0.2 | −2.0 / −2.4 |
| | 4 | 50.3 / 61.3 | −0.2 / −0.2 | −1.3 / −1.8 |
| | 8 | 47.6 / 58.2 | −0.2 / −0.1 | −0.7 / −0.9 |

- **순차화 단독은 손해**. rand 절대값이 47.6(all) → 50.0(seq)으로 +2.4. arm과 무관하게 모두가 겪음
- **재채점+순차화는 이득 누적**. all·seq에서 −1.5 → −0.3~−0.5로 줄던 oracle이 seq_rescore에선 −1.5 → −3.9로 커짐
- **그런데 재채점 단독은 별 효과 없음**. rescore의 oracle이 −0.5 → −0.7로 제자리
- **mag_diff는 방식을 안 가림**. n_rots=8에서 −1.5(all → rescore)와 −1.7(seq → seq_rescore)로 거의 같음
- **한꺼번에 학습하면 arm 간 차이가 좁혀짐**. seq_rescore는 oracle −3.9, mag_diff −0.4로 3.5 벌어지는데 rescore는 −0.7, −0.2로 0.5뿐. 두 sched는 pair가 같고 theta 학습 방식만 다름

신기한 결과. 예상과 달리 재채점 자체로는 큰 효력이 없고, 순차화와 같이 둬야 터짐

아마 rescore가 따지고 보면 최적해를 내는 방식이 아니라서 그런 것일지도. pair는 seq_rescore 궤적에서 고른 것인데 theta는 처음부터 다시 학습하니, 정작 자기 궤적에는 안 맞는 pair를 쓰는 셈

그리고 지금까지 alpha는 pre로 고정했는데, 현행과 비교하려면 joint여야 하므로 한 번 가드 풀고 확인

### joint 순차 loss

[q_proj layer=10/20 n_rots=8 alpha=joint sched=seq_rescore](runs/loss_260816-015024.log). all rand (현행) 대비 편차, all rand만 절대값. all rand는 [n_rots별 rng](#n_rots별-rng)로부터. rand는 seed 평균

| model | layer | all rand (절대) | seq_rescore rand | seq_rescore oracle |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 10 | 31.2 | +1.8 | **−1.5** |
| | 20 | 31.9 | +2.4 | **−1.5** |
| Llama-3.1-8B | 10 | 52.5 | +2.7 | **−1.6** |
| | 20 | 59.5 | +2.6 | **−1.8** |
| Mistral-7B-v0.3 | 10 | 38.7 | +2.2 | **−0.8** |
| | 20 | 53.1 | +2.7 | **−1.4** |
| gemma-2-9b | 10 | 44.6 | +1.7 | **−1.6** |
| | 20 | 42.4 | +1.0 | **−2.3** |
| 평균 | | 44.2 | +2.1 | **−1.6** |

- **8셀 전부 현행을 넘음**. −0.8~−2.3, 평균 −1.6. rand의 seed 폭이 0.1~0.4이므로 노이즈의 4~10배이고 부호가 뒤집히는 셀 없음

현행을 넘는 것까지 확인. 다만 더 이어가기에는 좀 지쳐서 여기까지만 하고 정리...

## 정리

### 저자 주장에 대해

- ~~independent rotations is highly robust to the pair selection strategy~~ → arm에 따라 −0.8~+2.6으로 벌어짐. 무관하지 않음
- ~~outlier와 normal channel을 짝지으면 더 효과적~~ → 논문 기준인 mag_diff가 n_rots=8에서 +2.5로 오히려 뒤쪽
- random pairing is nearly optimal → 현행 알고리즘 안에서는 맞음. 다만 무작위가 좋아서가 아니라 theta를 한꺼번에 학습하는 현향 구조에서는 rand보다 나아질 수 없기 때문

### 결론

- 손해를 내는 것은 정렬 자체가 아닌 대리지표
- rot마다 다시 채점하고 theta도 순차로 학습하면 현행을 넘음
- 다만 그 이득은 매우 미미함
- 무작위성·dummy·연결성·최적화·pair 고갈·점수 붕괴는 기각

### 한계

- q_proj 하나, layer 10/20 둘, 4bit·grp_size 128 하나. calibration도 고정
- 학습 비용이 n_rots배. all은 200스텝인데 seq_rescore는 rot마다 200스텝이라 1600스텝, 여기에 oracle 채점이 rot마다 붙음
- loss는 linear 하나의 출력 오차. end-to-end ppl까지는 확인 안 했음
- oracle은 rng를 각도 16개 그리드로 재는 근사. 진짜 최적이 아님
- joint 순차는 rand·oracle만 돌림. 대리지표가 거기서도 못 넘는지는 확인 안 했음

### 남은 것

- corr만 방향이 반대인 이유
- 크기로 정렬하는 둘만 rot 전 alpha에서 불리하게 출발하는 이유
- layer 0만 양상이 다른 이유
