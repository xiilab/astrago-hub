import torch


def get_modules(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2
        import torchvision.tv_tensors
        import v2_extras

        return torchvision.transforms.v2, torchvision.tv_tensors, v2_extras
    else:
        import transforms

        return transforms, None, None


class SegmentationPresetTrain:
    def __init__(
        self,
        *,
        imgsz,
        crop_size,
        hflip_prob=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        backend="pil",
        use_v2=False,
    ):
        T, tv_tensors, v2_extras = get_modules(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tv_tensor":
            transforms.append(T.ToImage())
        elif backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        if hflip_prob > 0:
            transforms += [T.RandomHorizontalFlip(hflip_prob)]

        if use_v2:
            # We need a custom pad transform here, since the padding we want to perform here is fundamentally
            # different from the padding in `RandomCrop` if `pad_if_needed=True`.
            transforms += [v2_extras.PadIfSmaller(crop_size, fill={tv_tensors.Mask: 255, "others": 0})]

        transforms += [T.RandomCrop(crop_size)]
        
        transforms += [T.Resize(size=(imgsz, imgsz))]

        if backend == "pil":
            transforms += [T.PILToTensor()]

        if use_v2:
            img_type = tv_tensors.Image if backend == "tv_tensor" else torch.Tensor
            transforms += [
                T.ToDtype(dtype={img_type: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True)
            ]
        else:
            # No need to explicitly convert masks as they're magically int64 already
            transforms += [T.ToDtype(torch.float, scale=True)]

        transforms += [T.Normalize(mean=mean, std=std)]
        if use_v2:
            transforms += [T.ToPureTensor()]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(
        self, *, imgsz, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), backend="pil", use_v2=False
    ):
        T, _, _ = get_modules(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms += [T.PILToTensor()]
        elif backend == "tv_tensor":
            transforms += [T.ToImage()]
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        # if use_v2:
        transforms += [T.Resize(size=(imgsz, imgsz))]
        # else:
        #     transforms += [T.RandomResize(min_size=base_size, max_size=base_size)]

        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2?
            transforms += [T.ToImage() if use_v2 else T.PILToTensor()]

        transforms += [
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
        ]
        if use_v2:
            transforms += [T.ToPureTensor()]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)
