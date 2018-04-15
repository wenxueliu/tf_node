## Anchor

anchor 包含 grid 和 ssd 两种 anchor 生成方法

anchor_generator.py  : anchor 生成器的抽象函数

* feature_map : [height, width] anchor 生成的输入
* feature_map_shape : feature_map 的 shape
* aspect_ratios ：即 width/height = aspect_ratios
* scale :  高度和宽度同时缩放比例
* num_anchors_per_location = scales * aspect_ratios
* expected_num_anchors = (num_anchors_per_location * feature_map_shape[0] * feature_map_shape[1])


### GridAnchorGenerator

* scales :  高度和宽度同时缩放比例；[0.5, 2.0, 3.0]
* aspect_ratios : 宽度与高度的比；[0.5, 1, 2]
* base_anchor_size : 滑动窗口的尺寸, [height, width]
* anchor_stride : 滑动窗口每次滑动的步长； 默认 16
* anchor_offset : 滑动窗口的开始滑动的偏移； 默认 [0，0]
* grid_height : 高度方向滑动的次数
* grid_width  : 宽度方向滑动的次数

其中

1. scales 和 aspect_ratios 组合记录所有的缩放比例
2. grid_height * grid_width 记录了总共需要滑动的次数

整个实现就是将 base_anchor_size 的每一个子图，
从偏移 anchor_offset  每次滑动 anchor_stride 个像素
滑动 grid_height * grid_width 次，返回所有所有的滑动窗口

### 例子

给定

feature_map_shape = (2, 3) # 控制高度方向和宽度方向滑动的次数
scales = [1.0, 0.5]
aspect_ratios = [1.0, 4.0]
base_anchor_size = [16, 16] # 滑动窗口大小为 16 x 16 的正方形
grid_height, grid_width = feature_map_shape # 高度和宽度方向都滑动 2 次
anchor_stride = [4, 3]  # 高度每次滑动 4，宽度每次滑动 3
anchor_offset = [1, 2]  # 高度偏移为 1，宽度偏移为 2

1. 生成不同的比例组合

[(1.0, 1.0), (0.5, 1.0), (1.0, 4.0), (0.5, 4.0)]

scales 和  aspect_ratios 总共生成 4 中比例

2. 生成不同比例的 base_anchor

公式为

    heigh = scales / sqrt(aspect_ratios) * base_anchor_size[0]
    width = scales * sqrt(aspect_ratios) * base_anchor_size[1]

因此，实际高度和宽度的比例为

[(1.0, 1.0), (0.5, 0.5), (0.5, 2.0), (0.25, 1.0)]

然后 base_anchor 乘以各个不同的比例，得到各个不同比例的 base_anchor

[(16.0, 16.0), (8.0, 8.0), (8.0, 32.0), (4.0, 16.0)]

3. 对于每一种 anchor

从 anchor_offset 开始：

* 高度以 anchor_stride[0] 的步长滑动 grid_height 次
* 宽度以 anchor_stride[1] 的步长滑动 grid_width 次

总共滑动  6(grid_height * grid_width) 次，分别为

[(1, 2), (1, 5), (1, 8), (5, 2), (5, 5), (5, 8)]

所有的 base_anchor 总共得到 24 (4*6) 个 anchor

需要注意的是, base_anchor 指的是高度和宽度，生成的 anchor 以 [0, 0] 为中心
[x_min, y_min, x_max, y_max]

比如 [16.0, 16.0] 滑动 [1,2] 之后，实际为

[-8 + 1, -8 + 2, 8 + 1, 8 + 2] 即 [-7, -6, 9, 10]，其他以此类推

[[ -7.  -6.   9.  10.]
 [ -3.  -2.   5.   6.]
 [ -3. -14.   5.  18.]
 [ -1.  -6.   3.  10.]
 [ -7.  -3.   9.  13.]
 [ -3.   1.   5.   9.]
 [ -3. -11.   5.  21.]
 [ -1.  -3.   3.  13.]
 [ -7.   0.   9.  16.]
 [ -3.   4.   5.  12.]
 [ -3.  -8.   5.  24.]
 [ -1.   0.   3.  16.]
 [ -3.  -6.  13.  10.]
 [  1.  -2.   9.   6.]
 [  1. -14.   9.  18.]
 [  3.  -6.   7.  10.]
 [ -3.  -3.  13.  13.]
 [  1.   1.   9.   9.]
 [  1. -11.   9.  21.]
 [  3.  -3.   7.  13.]
 [ -3.   0.  13.  16.]
 [  1.   4.   9.  12.]
 [  1.  -8.   9.  24.]
 [  3.   0.   7.  16.]]

注：这里的 anchor 为绝对坐标

详细可以参考附录

## 附录

```
#!/usr/bin/env python
# encoding: utf-8

import numpy as np

def tile_anchors(grid_height,
                 grid_width,
                 scales,
                 aspect_ratios,
                 base_anchor_size,
                 anchor_stride,
                 anchor_offset):

  # base_anchor_size 应用到 scales 和 aspect_ratios 中的所有变换，
  # 生成 scales * aspect_ratios 个子坐标
  # 注意变换方式
  ratio_sqrts = np.sqrt(aspect_ratios)
  heights = scales / ratio_sqrts * base_anchor_size[0]
  widths = scales * ratio_sqrts * base_anchor_size[1]
  print heights
  print widths

  y_centers = np.arange(grid_height)
  # 纵坐标要滑动的点
  y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
  x_centers = np.arange(grid_width)
  # 横坐标要滑动的点
  x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
  x_centers, y_centers = np.meshgrid(x_centers, y_centers)

  print zip(y_centers.reshape(-1), x_centers.reshape(-1))

  widths_grid, x_centers_grid = np.meshgrid(widths, x_centers)
  heights_grid, y_centers_grid = np.meshgrid(heights, y_centers)
  bbox_centers = np.stack([y_centers_grid, x_centers_grid], axis=2)
  bbox_sizes = np.stack([heights_grid, widths_grid], axis=2)
  # 滑动的点
  bbox_centers = np.reshape(bbox_centers, [-1, 2])
  print bbox_centers
  # 所有子图
  bbox_sizes = np.reshape(bbox_sizes, [-1, 2])
  # 将每一个子图，按照 bbox_centers 进行滑动后的坐标集合，
  # 即所有滑动窗口的坐标集合，格式我 [y_min, x_min, y_max, x_max]
  bbox_corners = np.concatenate([bbox_centers - .5 * bbox_sizes, bbox_centers + .5 * bbox_sizes], 1)
  print bbox_corners


def py_test():
  base_anchor_size = [16, 16]
  grid_height, grid_width = (2,3)
  anchor_stride = [4, 3]
  anchor_offset = [1, 2]
  scales = [1.0, 0.5]
  aspect_ratios = [1.0, 4.0]

  # 列出 scales, aspect_ratios 的所有组合
  scales_grid, aspect_ratios_grid = np.meshgrid(scales, aspect_ratios)
  scales_grid = np.reshape(scales_grid, [-1])
  aspect_ratios_grid = np.reshape(aspect_ratios_grid, [-1])

  tile_anchors(grid_height, grid_width,
            scales_grid,
            aspect_ratios_grid,
            base_anchor_size,
            anchor_stride,
            anchor_offset)

py_test()
```

输出

    [[1 2]
     [1 5]
     [5 2]
     [5 5]]
    [[ -7.  -6.   9.  10.]
     [ -7.  -3.   9.  13.]
     [ -3.  -6.  13.  10.]
     [ -3.  -3.  13.  13.]]
