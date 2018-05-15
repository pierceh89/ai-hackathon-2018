# 네이버 해커톤 2018

[네이버 AI 해커톤 2018](https://github.com/naver/ai-hackathon-2018)

Vamos 팀으로 참가하였고, 사정상 최종 라운드에는 참가하지 못했습니다. (라운드2 34등 최종스코어 0.85)

## 지식인 모델

세가지 모델을 테스트해봤습니다.
- 6 layer neural network
- [CNN for Sentence Classification](https://arxiv.org/abs/1408.5882)을 참조하고 input을 w2v으로 임베딩하여 사용
- [CNN for Sentence Classification](https://arxiv.org/abs/1408.5882)을 참조하고 input을 기본 c2v을 그대로 사용

### w2v 사용한 CNN 모델

- [Twitter Tokenizer](http://konlpy.org/en/v0.4.4/api/konlpy.tag/#module-konlpy.tag._twitter)를 이용해서 토큰을 추출한 뒤 w2v을 사용해서 워드 임베딩을 시행
- 트레이닝 때 출현하지 않은 단어를 UNKNOWN(영벡터)로 정의하고 구현
- over-fitting이 심해서 결과는 좋지 않았습니다.

이 모델에 한계를 나름대로 분석해보면
1. Corpus를 사용하지 않은 word-embedding
2. 출현하지 않은 단어에 대해 영벡터로 정의
3. 2 channel 사용하여 word-embedding까지 업데이트하지 않음
이라고 생각합니다.

### c2v (default) 사용한 CNN 모델

- 기본 c2v 임베딩에 CNN 구현

기존에 비해 5% 정확도가 상승했습니다.

### 그 외

- Xavier init, He init
- Data size doubling (텍스트 순서 스왑)

## 영화 리뷰

영화 리뷰는 tensorflow API로 포팅만 하고 특별한 작업은 하지 않았습니다.