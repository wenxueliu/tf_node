

图片表示都是 (height, width) 的顺序，不可颠倒，当将一个正方形用 aspect_ratios 变形的时候

    height = height / sqrt(aspect_ratios)
    width = width * (aspect_ratios)



## FasterRCNNMetaArch

## Anchor

参考 anchor.md

## boxCoder

## Feature Map

### NAS

FasterRCNNNASFeatureExtractor

#### Tips

    假设要生成两个向量的组合
    a, b
    x, y = np.meshgrid(a, b)
    x = x.reshape(-1)
    y = y.reshape(-1)

