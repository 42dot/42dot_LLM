# 챗베이커 (ChatBaker)

챗베이커 (ChatBaker)는 42dot에서 개발한 생성형 언어 모델입니다.
- 대한민국 최초의 한영통합 거대 언어 모델 (Large Language Model, LLM) 학습
- 한영통합 PLM을 기반으로 생성형 언어 모델 학습
- 자체 구축한 (수집, 정제) 데이터, 학습 인프라 사용

### [온라인 데모](demolink)
ChatBaker 한영통합 7B 모델을 경험해보세요!

[데모 샘플 GIF 추가]


## 생성형 언어 모델
챗베이커 (ChatBaker) 학습에 [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)의 베이스 코드인 [FastChat](https://github.com/lm-sys/FastChat)을 사용했고, 사용한 파라미터는 아래와 같습니다.

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay | Warmup ratio |
| -- | -- | -- | -- | -- | -- | -- |
| ChatBaker | 16 | 2e-5 | 3/6/9 | 2,048 | 0 | 0.03 |

A100 80G GPU 8장을 학습에 사용했습니다.

| Model | ChatBaker-1.3B-kr | ChatBaker-1.3B-kr-en | ChatBaker-7B-kr-en |
| -- | -- | -- | -- |
| Training time | 9 hours | 20 hours | 48 hours |

### 학습 데이터셋

요청 및 이에 대한 응답으로 이루어진 대화형태의 데이터를 사용했습니다.
- 한국어: 약 15만 건
- 영어: 약 25만 건

주) 챗베이커 (ChatBaker) 학습에 사용한 데이터는 공개하지 않습니다. 대신, 다양한 한국어 ([evolve-instruct](https://github.com/lcw99/evolve-instruct), [ko-lima-vicuna](https://huggingface.co/datasets/changpt/ko-lima-vicuna), 등) 및 영어 (ShareGPT, OpenAssistant, etc.) 대화 데이터가 공개되어 있습니다.

### 평가
- 비교대상:
  - [Polyglot-Ko-1.3B-SFT]: [Polyglot-Ko-1.3B](https://huggingface.co/EleutherAI/polyglot-ko-1.3b) 모델에 ChatBaker와 동일한 데이터로 학습한 모델
  - [ChatGPT](https://chat.openai.com/): OpenAI가 공개한 생성형 언어 모델 서비스
  - [Bard](https://bard.google.com/): Google이 공개한 생성형 언어 모델 서비스
- 평가 데이터셋:
[데이터셋 내용 추가]
- 평가 방법:
[평가 방법 추가]


<img src="asset/image.png" width="90%" height="90%"/>

[성능 그래프 대체]

#### 요약
[평가 결과 추가]


## 사전 학습 모델 (PLM)
### 아키텍쳐
Transformer decoder 기반의 [LLaMA](https://arxiv.org/abs/2302.13971) 아키텍쳐를 사용했고, 사용한 파라미터는 아래와 같습니다.

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| -- | -- | -- | -- | -- | -- |
| PLM | | | | 2,048 | |

| Hyperparameter | Layers | Attention heads | Hidden size |  |  |
| -- | -- | -- | -- | -- | -- |
| 1.3B | 24 | 32 | 2,048 | | |
| 7B | | | | | |

A100 80G GPU 256장 (8 GPUs * 32 Nodes)을 사용했습니다.

| Model | KO 1.3B | KOEN 1.3B | KOEN 7B |
| -- | -- | -- | -- |
| Training time | xx days | xx days | 30 days+ |


### 학습 데이터셋
- 한국어: 100B 토큰
- 영어: 1T 토큰

### 토크나이저
Byte-level BPE 토크나이저를 사용했고, 한국어와 한영통합 토크나이저는 각각 문서 100만건으로 학습했습니다.

### 평가
#### 한국어
- 비교대상:
  - [Polyglot-Ko](https://github.com/EleutherAI/polyglot): LLaMA 아키테쳐를 기반으로 한국어 213B 토큰 (863 GB)의 데이터셋으로 학습한 모델
  - [KoGPT2](https://github.com/SKT-AI/KoGPT2): GPT 아키텍쳐를 기반으로 40GB 이상의 한국어 데이터셋으로 학습한 모델
- 평가 데이터셋:
  - [KoBEST](https://huggingface.co/datasets/skt/kobest_v1) 
  - HyperClova에서 평가한 데이터셋은?
- 지표: Macro F1

[성능 그래프 추가]

#### 영어
- 비교대상:
  - [OPT](OPT 링크): 
  - [MPT](MPT 링크): 
  - [LLaMA](LLaMA 링크): 
- 평가 데이터셋: 영어 Benchmarks 14종
    - anli, arc, boolq, hellaswag, openbookqa, piqa, record, rte, truthfulqa_mc, wic, winogrande

[성능 그래프 추가]

#### 요약
- 한국어 모델은 polyglot-ko 1.3B 대비 약 5% 높은 성능 달성
- 한영통합 모델은 kogpt (skt) 대비 약 2% 높은 성능과, polyglot-ko 1.3B 대비 오차범위 이내의 성능을 보였습니다.

### 모델 공개

🤗[한영통합-1.3B](허깅페이스 링크)


## 한계점
다른 LLM (ChatGPT, Vicuna, 등)과 마찬가지로 챗베이커 (ChatBaker) 또한 많은 한계를 가지고 있습니다. 

We have noticed that, similar to other large language models, Vicuna has certain limitations. For instance, it is not good at tasks involving reasoning or mathematics, and it may have limitations in accurately identifying itself or ensuring the factual accuracy of its outputs. Additionally, it has not been sufficiently optimized to guarantee safety or mitigate potential toxicity or bias. To address the safety concerns, we use the OpenAI moderation API to filter out inappropriate user inputs in our online demo. Nonetheless, we anticipate that Vicuna can serve as an open starting point for future research to tackle these limitations.

## 라이센스
- 코드: 
- 모델:


