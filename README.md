# README.md

## Data Folder Format

![Untitled](https://github.com/xiilab/astrago-hub/assets/161695779/7601e28d-86d5-4fa4-bb8c-0013c3cb6d94)


- train, val 폴더 이름은 반드시 train, val로 고정할 것
- —data-path 인자값에는 최상위 폴더(data) 경로만 명시하면 된다.

---

## USAGE

- —pretrained 인자값의 경우 명시하면 해당 모델에 대한 가중치를 자동으로 가져와 학습함
- 만약 특정 가중치로 학습하고 싶은 경우 —weights 인자값 사용
- 더 자세한 사용법은 torchvision 공식 Git 사이트 참고
[https://github.com/pytorch/vision/tree/main/references/classification](https://github.com/pytorch/vision/tree/main/references/classification)

```
python train.py \
	--data-path ./data \
	--model resnet50 \
	--imgsz 640 \
	--batch-size 32 \
	--epochs 100 \
	--pretrained
```