## Data Folder Format

![Untitled](https://github.com/xiilab/astrago-hub/assets/161695779/056ea5d6-4d05-448a-b861-4a469c9e429e)

- train : train image 폴더
- val : val image 폴더
- 최상위 디렉토리(COCO)를 제외한 디렉토리 및 파일 이름은 위와 동일하게 세팅할 것
(train 및 val 하위 image 파일 이름은 상관없음)
- --data-dir 인자값엔 최상위 디렉토리(COCO) 경로만 넣으면 된다.

---

## USAGE

- YOLOX/exps/default에 모델 별 exp 설정 가능
(custom 예시는 YOLOX/exps/example/custom/yolox_s.py 참고)
- YOLOX/yolox/exp/yolox_base.py에서 직접 인자값 변경도 가능
- 자세한 사항은 아래 YOLOX Git 공식 페이지에서 확인
[https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

```
python tools/train.py \
	--exp_file exps/example/custom/yolox_s.py \  # 학습하고 싶은 모델 py 파일 경로 입력
	--imgsz 320 \
	--epoch 100 \
	--devices 1 \
	--batch-size 16 \
	--fp16 \
	-occupy \
	--ckpt pretrained/yolox_s.pth # 모델 가중치
```