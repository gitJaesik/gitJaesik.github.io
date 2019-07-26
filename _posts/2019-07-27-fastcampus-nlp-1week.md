---
layout: post
title: "NLP - 1week"
subtitle: "from fastcampus"
author: "Maguire1815"
header-img: "img/post-bg-infinity.jpg"
header-mask: 0.3
mathjax: true
tags:
  - nlp
  - al
  - ml
---

# 1 week

- Probability Function Approximation
  - Probability
  - Expectation and Sampling
  - Maximum Likelihood Estimation (MLE)
  - Information
- Dimension Reduction
  - Introduction
  - Principal Component Analysis (PCA)
  - Manifold Hypothesis
  - Auto-Encoder (AE)

- An Introduction to Deep Learning for Natural Language Processing
  - History of NLP
  - Paradigm Shift in NLP
  - Why NLP is difficult?
  - Why Korean NLP is Hell?

## Probability Function Approximation

### Probability

- 랜덤 변수 (Discrete vs Continuous로 구분분

#### Discrete Variable

- 주사위 예제
- Probability Mass Function이 존재 (PMF)

#### Continuous Variable

- *주의* : 면적을 보기 때문에, 특정 지점에 확률값이 1보다 클 수 있음
- Probability Density Function이 존재 (PDF)

NLP vs 이미지, 비전, ASR의 차이는 이산, 연속의 차이

#### Joint Distribution & Marginal Probability

- 2개의 Random Variable이 합쳐진 확률

#### Conditional Probability

- 조건부 확률 분포

#### Bayes Theorem

- Likelihood를 이용한 확률 구하기

#### Monty-hall Problem

- Joint Distribution & Marginal Probability, Conditional Probability, Bayes Theorem를 이용해서 구할 수 있다.

#### Probability Distribution

- Discrete
  - Bernoulli : 이항 분포
  - Multinoulli
- Continuous
  - Uniform
  - Gaussian : 정규분포

### Expectation and Sampling
 
#### Expectation

- 기대값 = 확률 * 보상

#### Monte Carlo Sampling

- Sampling을 통해서 적분

### Maximum Likelihood Estimation

#### Machine Learning

- 내가 보지 못한 데이터에서 잘 예측하는 것
 
#### Likelihood as Function

- 일어날 것 같은 일을 측정
  - Probability in discrete variable
    Probability Density in continuous variable

#### MLE Example

- K ~ B(n, seta)
  - 결국 seta에 대한 함수로 변경 됨

#### MLE Gradient Ascent

- Ground Truth를 구한다.

#### Log Likelihood

- Prevent underflow
- 곱보다 합이 빠름
- Remove exponent on Gaussian

#### Negative Log Likelihood

### Information

#### Neural Network is

- 확률분포함수다

#### If we have a dataset

- approximate the real-distribution

#### Kullback-Leibler Divergence

- 분포 얘기
- 두 분포간의 괴리, divergence를 설명

#### Information

- 불확실성 & 1/확률 & 정보량

#### Entropy

- 정보의 기대값(expectation)
- 얼마나 플랫? 샤프? 한지

#### Cross-Entropy

- KL divergence처럼 분포간의 비슷한 정도를 구할 수 있다.

#### Relationship with MLE

- Cross-Entropy = Likelihood

#### Relationship with KL Divergence

- Cross-Entropy = KL D

#### MSE Loss

- 유클리드한 distance minimazation

## Dimension Reduction

### Introduction

#### Hyperplane

- 스칼라, 벡터메트릭스, 텐서-

#### Curse of Dimensionality

- sparse!
- more difficult

#### Need of Dimension Reduction

- 비슷한 정보 지우기

#### Why Deep Learning

- non-linear한 차원 축소가 가능하기 때문

### Principal Component Analysis (PCA)

#### PCA

- 주성분 분석
- Maximize variance
- Minimize residuals

### Manifold Hypothesis

#### Manifold Hypothesis

- 초평면
- 실제로는 증명 못하지만 있을 거라 가정하고 하면 좋은 성과

#### Decision Boundary

- binary classification

#### Manifold with Deep Learning

- finding manifold = train

### Auto-Encoder (AE)

#### AE

- 입력 받은 거 똑같이 밷는 것 === Seq to Seq, Sequential 한 데이터를 다루는 Auto Encode
- 가장 차원이 낮은 부분이 bottle leck이 됨 = 이곳의 manifold를 찾자
- Encoder = Dimension Reduction
- Decoder = Reconstruction

#### Projection to Manifold Surface

- Manifold하면 다시 못오는 다대일 함수

#### Reconstruction from Manifold Surface

- Reconstruction Error : MSE Loss
- Train Objective : Minimize Reconstruction Errors

#### Add Noise to avoid Overfitting

## An Introduction to Deep Learning for Natural Language Processing

### Index

### History of NLP

#### History of NLP

- Seq2Seq, Attention 이전에는 text -> vector밖에 불가능

#### Neural Machine Translation

- 2014년 Sequence-to-sequence(seq2seq)가 소개
- Traditional SMT <- 통계 기반 번역 <- corpus가 작을 때는 쓸만함

### Paradigm Shift in NLP

#### Paradigm Shift in NLP from Traditional to Deep Learning

- 여러단계의 sub-module
- 무겁고 복잡
- error propagation
- end-to-end model들로 대체

#### 접근 방법의 변화

- Traditional NLP
  - lang = discrete
  - 보고 해석하기는 쉬움

- NLP with Deep Learning
  - 단어를 continuous 한 vector로 나타냄

#### NLP System with Deep Learning

- 이산적인 심볼릭 데이터
  - 사람이 이해
  - 입력 : x
  - 출력 : y
- 연속적인 데이터
  - 연산 효율 높음
  - 입력 : hx
  - 출력 : hy

### Why NLP is difficult?

#### Ambiguity

- 중의성 해소 (word sence disambiguation)
- 문장 내 정보의 부족

#### Paraphrase

- 문장의 표현 형식은 다양하고, 비슷한 의미의 단어들이 존재하기 때문

#### Discrete, Not Continuous

- continuous한 값으로 바꿔야 함

#### Noise and Normalization

- data 손실 우려

### Why Korean NLP is Hell?

#### 교착어

- 어간에 접사가 붙어 단어를 이루고 의미와 문법적 기능이 정해짐

#### 띄어쓰기

- 표준이 계속 바뀜
- 없더라도 해석 가능
- 띄어쓰기를 정제(normalization) 해주는 process가 필요

#### 평서문과 의문문의 차이

- 물음표

#### 주어 생력

- 주어 없음

#### 한자 기반의 언어

- 뜻 많음
