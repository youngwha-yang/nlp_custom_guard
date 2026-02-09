# 🛡️ Semantic Intent-based Guardrail
> **Overcoming Regex Limitations with Contextual NLP Intelligence**

본 저장소는 정규표현식(Regex)의 한계를 극복하고, 사내 특수 SW 환경에 최적화된 유해 문구 차단 시스템을 구축하기 위한 **KoELECTRA 기반 가드레일** 모델의 학습 및 테스트 코드를 포함하고 있습니다.

---

## 📁 Repository Structure

```text
.
├── dataset/
│   └── total_train_v3.csv       # Synthetic Data + Hard-Negative 데이터셋
├── model/
│   └── (trained_model_files)    # 최종 학습 완료된 KoELECTRA v3 모델 가중치 및 설정
├── 1_GenerateSyntheticData.ipynb # GPT-4o 활용 데이터 생성 및 전처리 노트북
└── 2_Training.ipynb             # 모델 학습, 성능 평가 및 유사도 분석 노트북
```
## 🚀 Getting Started

### 1. Data Generation ([1_GenerateSyntheticData.ipynb](./1_GenerateSyntheticData.ipynb))
* **목적**: 보안 위협 데이터의 희소성을 해결하기 위해 GPT-4o를 활용한 합성 데이터(Synthetic Data)를 생성합니다.
* **주요 기능**: 
    * 난독화 기법(특수문자 삽입 등)이 적용된 변칙 공격 시나리오 생성
    * **Hard-Negative**(위험 키워드를 포함한 정상 문장) 데이터셋 구축을 통한 변별력 확보

### 2. Model Training & Evaluation ([2_Training.ipynb](./2_Training.ipynb))
* **목적**: `KoELECTRA-small-v3` 모델을 미세 조정(Fine-tuning)하고 성능을 검증합니다.
* **핵심 과정**:
    * **Training**: 정밀한 Hyper-parameter 튜닝(LR: 1e-5, Epoch: 40)을 통한 문맥 학습
    * **Evaluation**: 정확도(Accuracy) 100% 및 AUC 1.0 달성 확인
    * **Error Analysis**: 테스트 샘플과 학습 데이터 간 **Cosine Similarity** 분석을 통한 데이터 커버리지 진단

---

## 🧠 Core Strategy

### Why KoELECTRA?
* ELECTRA의 **Discriminator** 구조는 '가짜 단어'를 판별하도록 설계되어 있어, 문장의 유해성을 이진 분류하는 가드레일 태스크에 BERT 계열보다 높은 효율성과 정확도를 보입니다.
* 실시간 보안 필터링이 필요한 환경에서 요구되는 경량성과 빠른 추론 속도를 보장합니다.

### Hard-Negative Mining
* 단순히 공격 패턴만 학습하는 것이 아니라, `class=null`과 같은 키워드가 포함된 **정상적인 코딩 질문**을 함께 학습시켰습니다.
* 이를 통해 모델이 특정 키워드에 매몰되지 않고(Keyword Bias 제거), 정교한 **결정 경계(Decision Boundary)**를 형성하여 문맥적 의도를 파악하도록 설계했습니다.

---

## 📈 Performance
* **Accuracy**: **100%** (Test set 32개 샘플 전수 정답)
* **AUC Score**: **1.00** (완벽한 클래스 변별력 증명)
* **Confidence Analysis**: 
    * 공격(Label 1) 샘플 확신도: 평균 **0.71 ~ 0.72**
    * 정상(Label 0) 샘플 확신도: 평균 **0.67 ~ 0.70**
    * 두 클래스 간 일관된 수치적 격차를 확보하여 안정적인 차단 임계값(Threshold) 설정 가능 확인

---

## 🛠️ Tech Stack
* **Language**: Python 3.10+
* **Environment**: Google Colab / Jupyter Notebook
* **Main Libraries**:
    * `transformers`: KoELECTRA 모델 로드 및 Fine-tuning
    * `scikit-learn`: AUC 측정 및 TF-IDF 유사도 분석
    * `pandas` & `numpy`: 데이터셋 관리 및 수치 계산
    * `torch`: 딥러닝 프레임워크 기반 학습 수행
* **Base Model**: `monologg/koelectra-small-v3-discriminator`
