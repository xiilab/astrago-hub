# README.md

## 소개

![Untitled](README%20md%203bf15a4a9c1f46b4bd1ec121482dbe9c/Untitled.png)

UNet은 이미지 세그멘테이션(이미지 내의 픽셀 수준에서 객체 경계를 식별하는 작업)을 위한 딥러닝 아키텍처 중 하나입니다. UNet은 2015년에 발표된 "U-Net: Convolutional Networks for Biomedical Image Segmentation" 논문에서 처음으로 제안되었습니다.

UNet은 Fully Convolutional Network(FCN) 아키텍처의 한 종류로, 입력 이미지에서부터 출력 이미지(세그멘테이션 마스크)를 생성하기 위해 컨볼루션 레이어와 업샘플링(upsampling) 레이어로 구성됩니다. 특히 UNet은 특징 추출을 위한 다운샘플링(Downsampling) 경로와 위치 정보를 복원하기 위한 업샘플링(Upsampling) 경로로 이루어진 대칭 구조를 가지고 있습니다.

UNet은 의료 이미지 분야에서 주로 사용되어 왔으며, 세포 이미지, 혈관 이미지, 조직 이미지 등 다양한 의료 영상에서 객체(세포, 조직, 혈관 등)를 정확하게 분할하는 데 효과적으로 활용됩니다. 그러나 최근에는 UNet의 아이디어가 다른 분야로 확장되어 이미지 세그멘테이션 이외의 작업에도 적용되고 있습니다. UNet의 간결하고 강력한 아키텍처는 다양한 응용 분야에서 성능을 입증하고 있습니다.

---

## 폴더 구조 및 설명

![Untitled](README%20md%203bf15a4a9c1f46b4bd1ec121482dbe9c/Untitled%201.png)

- **data**
    - 데이터 폴더에는 'imgs'와 'masks'라는 두 개의 하위 폴더가 있습니다. 이들 폴더는 각각 원본 이미지와 해당하는 마스크 이미지를 저장합니다.
    - imgs : 원본 이미지를 포함합니다.
    - masks : 원본 이미지에 해당하는 마스크 이미지를 포함합니다.
- **checkpoints**
    - 매 epoch마다의 가중치 파일이 저장됩니다.
- **train_results**
    - 학습 결과 best val score을 가진 가중치가 best.pt라는 이름으로 저장됩니다.
- **predict_results**
    - 예측 결과 이미지가 저장됩니다.

<aside>
💡 utils 디렉토리 하위에 astrago.py 파일이 있습니다.

</aside>

---

## 사용법

- **Train**
    
    ```bash
    python train.py \
    		--data_dir ./data \
        --epochs 100 \
        --batch-size 16 \
        --imgsz 320 \
        --validation 50.0 \
        --learning-rate 0.0001 \
        --classes 2
    ```
    
- **Predict**
    
    ```bash
    python predict.py \
    		--model ./train_results/run/best.pth \
    		--input ./test_data \
    		--mask-threshold 0.45 \
    		--imgsz 320 \
    		--classes 2
    ```