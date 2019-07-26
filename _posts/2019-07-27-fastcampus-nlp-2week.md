# 2 week

- An Introduction To Deep Learning for Natural Language Processing
  - Recent Trends in NLG

- Process Overview
  1. 말뭉치(corpus) 수집
  2. 말뭉치(corpus) 정제(normalization)
    a. Cleaning (de-noising)
    b. Sentence Tokenization
    c. Tokenization
    d. Subword segmentation --optional
  3. 양방향(bi-lingual) 망뭉치(corpus) 정렬(align) - optional
  4. Raw corpus에 대해 align 이후에 다시 2번 과정 반복
- Language Model
  - Introduction
  - Equations
  - Applications
  - n-gram
  - Smoothing and Discounting
  - Evaluation
  - Neural Language Modeling

## review

- neural network는 선형적인 pca와 다르게 비선형적인 manifold와 같은 방식으로 진행해서 학습이 더 잘됨

- nlp 역사
  - discrete한 속성을 바꾸지 않고 연구하던 nlp를 continuous하게 변경하여 연구함
    - discrete와 continuous의 가장 큰 차이는 확률을 알 수 있음 vs 없음
    - discrete한 것을 continuous하게 바꾸면 정보의 손실이 있을 수 있음
  - nlp의 모호성
    - 사람의 머리에서는 discrete하지 않지만 사용하는 정보는 discrete
  - 한국어 (교착어)
    - 띄어쓰기 개판
    - 어순 개판
    - 효율성이 높음 (너도 알고 나도 아는 정보는 말하지 않음) <-> 컴퓨터 입장에서는 너만 알고 나는 모르는 것
  - Language Model
    - 한국어 : 접사만 잘 붙이면 어떤 단어 뒤에 어떤 단어가 나오든 상관 없음
    - 영어 : 어순이 중요 (스페인어 더 심함)

## Recent Trends in NLG

### RNNLM

- 기존: 통계기반
- RNN을 사용해서 Language Model을 구현
  - 언어모델 자체는 좋음
  - 2개 섞으면 더 좋음 (interpolation 같은)
  - End to End로 넘어가고 있지만 아직은 WFS를 사용하고 있음
- n-gram은 자기가 본거만 됨, RNN은 안 본 것도 됨
- 기계번역은 구조적인 한계로 WST를 못함

### Word2Vec

- 단순한 구조의 neural network를 사용하여 효과적으로 단어들을 hyper plane에 성공적으로 projection
- 당시에는 속도도 빠르고, 임베딩도 잘되는 것 (지금은 속도가 느리고, 임베딩도 안되는 것)
- 당시에는 혁명적

### CNN on NLP

- 획기적
- 문장은 단어들의 집합, 문장은 가변길이 -> 당연히 RNN을 써야지를 뒤집음
- 그러나 갑자기 CNN을 썼고 잘되니까
- Word Embeding을 통해서 Pattern을 잘 분석 (CNN은 원래 pattern을 잘 분석)

### Seqeuence-to-Sequence

- 문장을 숫자로, **숫자를 문장으로**
- 숫자를 문장으로 만드는게 핵심
- 옛날에는 문장을 숫자로 바꿔서 분류만 했었음

### Attention

- 기존의 한계를 돌파
  - RNN을 뛰어넘어 LSTM이 좋다고 했지만 무한대로 기억할 수 없음
  - Attetion을 쓰면 상관없이 시간을 뛰어넘어서 접근을 함
  - Seq2Seq로 문장을 생성하면서 Attetion이 기울기가 long dependence로 큰 혁명

### Memory Augmented Neural Network (MANN)

- Attention 성공 후 나온 방법

| Attention        | MANN     |
|------------------|----------|
| read or not read | read 20% |
| read only        | CRUD     |

### NLP and Reinforcement Learning

- Computer Vision
  - 기존 이미지를 recognation에 관심이 있었음
  - generative learning에 관심이 생김
- GAN을 사용하는 이유 : MSE의 한계를 돌파하기 위해
- Cross-Entropy, 기계번역의 objective와 훈련에 사용하는 cross-entropy에는 괴리가 있다. (순서가 중요 : cross-entropy)
- NLP는 generative learning을 원래 하고 있었음
  - NLP에서 GAN은 불가능해서 NLG로 감 (강화학습을 사용)

### Transfer Learning

- bert가 대표전
- GPU 엄청 사서, DATA 엄청 모아서 때려 넣으니 성능이 좋아짐
  - 실무자 입장에서는 좋음
  - text classification을 하려면 bert 다운 받아서 하면 된다.
- gpt2
  - corpus를 엄청 모아서 사람이 분류를 하고 유래가 없던 큰 corpus를 구축함 (open ai)
  - 심지어 아직 underfitting이라는 엄청난 데이터

## Preprocessing

### NLP 전처리의 중요성

- 요즘은 기술력의 차이가 거의 없음
  - 알고리즘 뻔함
  - 기계번역 사용에서 쓰이는거 대부분 transfer learning으로 들어감
  - 약간의 variation은 있음
  - 그래서 데이터가 중요하게 됨
- 즉, 전처리가 중요함
- 수학적인 지식보다 언어적인 지식이 더 필요한 분야

### Corpus

- Monolingual Corpus
- Bi-lingual Corpus
- Multi-lingual Corpus
  - Parallel Corpus

### Process Overview

- 문제를 정의하고 수집해야하는 말뭉치를 수집
  - 음성인식
    - 괄호, 특수기호는 지움
  - 기계번역
    - 정제가 쉬움
    - 웬만한거 다 남겨도 됨
    I 괄호 다 남겨도됨

- Sentence Tokenization
  - corpus의 형태는 한문장에 한 라인에 들어가 있어야 함
- Tokenization
  - 교착어
  - 접사
- 띄어쓰기 normalization
  - 같은 단어의 띄어쓰기가 다르면 다른 단어가 됨
- bpe (압축 알고리즘)
  - sub word의 뜻을 해석

### Collecting Corpus

- 데이터를 훈련을 해서 만든 모델의 저작권이 어디에 있는가? still ambiguous
- 공인 데이터 사용
- Crawling - robot.txt에서 수집
  
#### robots.txt

- crawling의 법적인 시비에 참고

#### Monolingual Corpora

- corpus 어디에서 뽑아 올 수 있나?
  - 블로그
  - 뉴스 기사
  - wikipedia
  - 대화체
    - 채팅로그가 가장 필요 (챗봇을 기준으로)
  - 드라마 (CSI?)
  - Ted (parallel)
  - 신문사
  - 영화
    - you did it /아빠가 그랬자나
    - you did it /오빠가 그랬자나
    - 욕설
  - monolingual corpus를 어떻게 늘릴 것인가가 가장 큰 문제

### Normalization Corpus

#### Denoising

- 전각문자
  - 언어별로 특징이 있음

#### 대소문자 통일

- 대소문자
  - 통일 필요
  - NIC, nic

#### Regular Expression @ Python

- 전처리를 하기 위해 (시작과 끝)
- 딥러닝 이전에는 필수 - 1000번 학습 할 걸 250, 250, 250, 250씩 학습이 될 수 있음 (손해)
- 딥러닝 이후에는 임베딩이 잘되서 알아서 잘됨

#### Text Editor with regex

- 코딩해야 할 부분까지 치환 가능

#### Conclusion

- cleaning 작업이 반복의 연속
  - (규칙정하고 수행)X무한반복
  - 적당할 때 끊는 것이 중요
- 정제작업은 끝이 없음

#### Sentence Tokenization

- 한문장 in 한라인
- 원하는 형태로 정제를 해야함

### Multiple sentence/line

- nltk에 있음
  - 영어기반
  - appendix 참고

### Tokenization

#### Part of Speech Tagging, Tokenization (Segmentation)

- 진짜 원하는 Tokenization --> Segmentation
  - POS 형태소분석기 사용
- 띄어쓰기
  - Sparsity 감소
  - 한국어 이외 언어
    - 영어 : 원래 잘되있음
    - 중국어, 일본어 : 띄어쓰기 없음
- Mecab : 일본어로부터 왔음, 좋음
  - 일본어는 한국어와 다름, 언어의 뿌리가 같으려면 지명이 같아야 함
- KoNLPy : 속도가 느림
- Stanford Parser, PKU
  - 임의의 테스트셋에서 성능이 비슷
  - 편한거 쓰면 됨 --> Python (for si)
- 영어 (Moses with NLTK)

#### Tokenization Conclustion

- 어차피 쉬운거는 다 잘됨
- 어떠한 정책이 잘 맞는지 확인 필요
- 또는 SI가 중요

### 양방향(bi-lingual) 말뭉치(corpus) 정렬(align) - optional

#### Proces Overview of Parallel Corpus Alignment

- 번역기가 없는대 정렬 (모순)
- Champolion을 통해 문장 aligning
  - 사실 사전임 번역이 아님
- MUSE
  - Facebook에서 만듬
  - 자동으로 사전을 만들 수 있음

#### Word Translation Dictionary

- 만든 다음에 지우기

#### Align via Champollion

- Champollion의 이름 유래
  - 이집트 상형문자 해독자
- Align
  - 한국어 문장의 4번째와 영어의 3번째가 mapping인지 등 하는 것

### Raw corpus에 대해 align 이후에 다시 2번 과정 반복

#### Segmentation using Subword (Byte Pare Encoding, BPE)

- 최근에 도입된 방법
- 단어를 더 작게 쪼갬
- 쪼개는 방법이 어려웠으나 Count기반으로 하니 잘됨
- 장점
  - 잘개 쪼개면 어휘 수가 줄어듦
    - 캐릭터 단위로 쪼개면 어휘수가 글자 수
    - sparsity가 줄어 듦
  - 모르는 글자가 없음
    - 대학교 --> 대 학교 모르는 단어를 아는 단어로 조합
    - unknown이 없음
    - 버카충
      - 맘충, 급식충이 있었으면 조합해서 추측
- 형태소 분석기 한번 돌리고 BPE돌리면 좋음

#### Subword Segmentation Example

#### Detokenization

## Language Modeling

### Introduction

- 언어 모델
  - 문장 자체의 출현 확률을 예측
  - 많은 문장을 수집하여 확률을 계산 할 수 있음
  - 궁금적인 목표는 문자의 분포를 모델링 하는 것

#### Again, Korean is Hell

- 단어에 어순이 중요하지 않기 때문에, 생략 가능하기 때문에 확률을 계산하는게 힘듦
- sparseness가 높아짐

### Equations

#### Expression

- 조건부 확률 (Conditional Probaility)
  - P(A,B) = P(A)P(B|A)
- Chain Rule
  - P(A,B,C,D) = P(A)P(B|A)P(C|A,B)P(D|A,B,C)
- Language Model Equation
  - P(w1, w2, ..., wk) = 파이 i=1부터 i=k까지 P(wi|w < i )
  - w is word

#### Expressions Example

- P(BOS, 나는, 학교에, 갑니다, EOS)
- Begining of sentence, End of sentence
- 주의! - '갑니다'가 문장의 끝이 아님, 'EOS'가 끝
- 확률은 corpus에서 세면 됨
- P(갑니다|BOS, 나는, 학교에) = COUNT(BOS, 나는, 학교에, 값니다) / COUNT(BOS, 나는, 학교에)

### Applications

- Natural Language Generation

| Task                          | Description                                                                                         |
|-------------------------------|-----------------------------------------------------------------------------------------------------|
| Speech Recognition            | Acoustic Model과 결합, 인식된 phone(음소)의 sequence에 대해서 좀 더 높은 확률을 갖는 sequnce로 보완 |
| Machine Translation           | 번역 모델과 결합하여, 번역 된 결과 문장을 자연스럽게 만듬                                           |
| Optical Character Recognition | 인식된 character candidate sequence에 대해서 좀 더 높은 확률을 갖는 sequence를 선택하도록 도움      |
| Other NLG Tasks               | 뉴스 기사 생성, chat-bot등                                                                          |
| Other..                       | 검색어 자동 완성 등...                                                                              |

- Automatic Speech Recognition
  - Objective
    - Y = argmaxP(Y|X)  
  - end2end vs 기존
    - end2end에서는 학습하는 부분이 적음

#### Automatic Speech Recognition (ASR)

- By Bayes Theorem
- AM | LM

### n-gram

#### Sparseness

- 단어의 조합의 경우의 수는 무한대에 가까움
- 문장이 길면 Count를 구할 수 없음
  - 똑같은 단어의 조합이 rare하기 때문
  - 분자 or 분모가 0이 되어 버릴 수 있음

#### Markov Assumption

- 모든 단어를 볼 필요 없이 앞 k개의 단어만
- P(xi|x1, x2, ..., xi-1) = P(xi|xi-k,...,xi-1)
- 문장의 확률
  - P(x1, x2, ..., xi) = 문장의 확률의 곱
- Log를 취하여 덧셈으로 바꾼다.

#### n-gram

- k=0, 1-gram, uni-gram
- k=1, 2-gram, bi-gram
- k=3, 3-gram, tri-gram
- 보통 3-gram, 4-gram을 사용
- n이 커질 수록 다시 sparse 해
- n이 너무 작아질 수록 왜곡이 심해짐
- 직관적
- table을 만들어서 lookup으로 쿼리만 날려도 되는 부분은 좋음
  - 모바일에 올리기 좋음
  - 모바일 가변 음성 인식기에 좋음

### Smoothing and Discounting

#### Smoothing

- Training corpus에 없는 unseen word sequence의 확률은 0? 아니다
- Unseen word sequence에 대한 대처
  - Smoothing
  - Discounting
- Popular algorithm
  - Modified Kneser-Ney Discounting

#### Add one smoothing

- Add one to every n-gram => generalize Add k => m/v

#### Held-out corpus

- Trainset 6
- Development Set == Held-out Set 2
- Test set 2

#### Absolute Discounting

- 상수배 차이

#### Kneser-Ney Discounting

- A-D에서 발전한 것이 KN-D
- 다양한 단어 뒤에서 나타나는 단어일 수록 unseen word sequence에 등장할 확률이 높은
- 제일 많이 쓰이는 방법

#### Interpolation

- 두 개의 서로 다른 language model을 일정하게 섞어줌
- 일반적으로 general corpus lm + 특정 domain의 corpus lm을 합칠 때 사용
- general corpus와 domain corpus를 ratio(hyper parameter)를 잘 조합하여 합쳐야 함

#### Interpolation Example

- Count를 조정해서 확률을 조정

#### Back-off

- Back-off - 직역 : 물러나, 꺼저
- Markov assumption 처럼 n이 안되면 n을 줄여감

#### Back-off Example

- 음성인식 tri-gram 많이 사용

#### Conclusion (n-gram)

- Pros
  - 쉽다
- Cons
  - unseen word sequence에 대처 불가
  - generalization은 unseen의 반 비례
  - 용량이 exponetial하게 커짐
- conclusion
  - 문장이 좋은지 나쁜지만 판단? => 사용
  - 문장을 translation 한다 => 사용x Neural Network를 사용 해야 함
    - translation은 하지만 poor

### Evaluation

- Intrinsic evaluation (정성 평가)
  - 정확
  - 시간과 비용
- Extrinsic evaluation (정량평가)
  - 꼭 필요함
- Perplexity (PPL)
  - 문장의 확률로 평가
  - 확률이 높을 수록 PPL은 작아지며, 작은 값이 더 좋은 것

#### Perplexity

- 테스트 문장에 대해서 확률을 높게 예측할 수록 좋은 언어 모델
- 주사위
  - PPL(x) = (1/6^n)^-1/n = 6
  - 매 time-step마다 6개의 확률이 있음
- PPL이 100?
  - 매 타입마다 100가지의 가능성이 있음
  - 낮은게 좋음

#### Entropy and Perplexity

- Cross-Entropy를 exponential한 것이 PPL
- PPL minimization == Cross-Entropy minimization

### Neural Language Model

- Resolve Sparsity
  - Training set
  - Test set
    - n-gram은 못하는 unseen word sequence를 맞출 수 있음
- 더 이상 markov assumption을 안 써도 됨

### Auto-regressive and Teacher Forcing

- Language Model은 기본적으로 auto-regressive함
  - 과거의 자기 자신의 값을 참조하여 현재의 값을 추론
  - Autu-regressive 예
    - 주가
- 훈련 때는 정답을 넣음
  - x hat = argmax x P(xt|x < t; seta) where X = {x1, x2, ..., xn}
- Training Mode
- Inference Mode
- Language Model
  - Auto-regressive한 속성을 가지고 있음

#### Because of Teacher Forcing

- 틀린 것을 넣고 정답을 예측 할 수 없으므로 정답을 다음에 넣어줌

#### Conclusion

- Pros
  - 훨씬 긴 과거를 기억할 수 있음 (>n)
  - Better generalization: unseen word sequence에 효율적 대처 가능
  - 작은 memory 사용
- Cons
  - WFST와 결합이 어려워 기존 application에 사용의 한계 (ASR, SMT)
  - n-gram에 비해 구현이 어렵고 연산량이 많음
