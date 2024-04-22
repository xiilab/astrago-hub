# README.md

## 소개

![Untitled](https://github.com/xiilab/astrago-hub/assets/161695779/84556d7f-1ca7-430b-a8bf-3a83e359a0a3)

ResNeXT는 Microsoft Research에서 개발한 심층 인공 신경망 아키텍처 중 하나입니다. 이 아키텍처는 ResNet(Residual Network)의 발전된 형태로, ResNet의 핵심 개념을 기반으로 하면서도 네트워크의 성능을 더욱 향상시키는 방법을 제안합니다.

ResNeXT는 여러 개의 동일한 블록을 병렬로 연결하여 네트워크를 구성하는 아이디어를 가지고 있습니다. 이러한 병렬 블록 구조는 "카디널리티(cardinality)"라고 불리며, 네트워크가 다양한 특징을 학습하고 병렬로 처리할 수 있도록 합니다. 각 블록 내부에서는 일반적으로 ResNet과 유사한 구조인 Residual Unit을 사용하여 네트워크의 깊이를 확장하고 학습 가능한 파라미터 수를 늘립니다.

ResNeXT의 주요 특징은 다음과 같습니다:

1. **카디널리티(Cardinality)**: ResNeXT는 병렬로 연결된 여러 개의 작은 블록을 사용하여 네트워크를 구성합니다. 이러한 블록은 서로 다른 "카디널리티"를 가질 수 있으며, 이는 각 블록이 서로 다른 특징을 학습하고 결합할 수 있도록 합니다.
2. **깊이(Depth)**: ResNeXT는 깊은 신경망을 구성하여 복잡한 데이터 패턴을 학습할 수 있습니다. 이러한 깊이는 네트워크의 표현력을 향상시키고, 더 복잡한 문제에 대한 더 정확한 해결책을 제공합니다.
3. **성능(Performance)**: ResNeXT는 높은 정확도와 효율성을 제공하는 효과적인 심층 신경망 아키텍처입니다. ImageNet과 같은 대규모 이미지 데이터셋에서 다른 최신 신경망 아키텍처와 비교하여 우수한 성능을 보여주었습니다.

ResNeXT는 다양한 컴퓨터 비전 작업에 사용되며, 특히 이미지 분류, 객체 감지, 세분화 등의 작업에서 널리 사용됩니다.

---

## 폴더 구조 및 설명

![Untitled 1](https://github.com/xiilab/astrago-hub/assets/161695779/cb908398-e246-4b0b-9d37-b8dbb9ff3545)

- **data**
    - 데이터 폴더에는 'train'과 'val'이라는 두 개의 하위 폴더가 있습니다. 이들 폴더는 각각 학습 데이터, 검증 데이터 이미지를 각 class 폴더에 저장합니다.
- **train_results**
    - 학습 결과 best val score을 가진 가중치가 best_model.pth라는 이름으로 저장됩니다.

<aside>
💡 utils 디렉토리 하위에 astrago.py 파일이 있습니다.

</aside>

---

## 사용법

- **Train**
    
    ```bash
    python train.py \
    	--data_dir ./data \
        --epoch 100 \
        --batch 16 \
        --imgsz 320 \
        --lr 0.001 \
        --pretrained True \
        --save ./train_results/run
    ```