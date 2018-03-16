
## Anchor

aspect_ratios ：即 width/height = aspect_ratios
scale : 不同的缩放比例

### GridAnchorGenerator

* scales :  高度和宽度的缩放比例；[0.5, 2.0, 3.0]
* aspect_ratios : 宽度与高度的比；[0.5, 1, 2]
* base_anchor_size : 滑动窗口的尺寸, [height, width]
* anchor_stride : 滑动窗口每次滑动的步长；
* anchor_offset : 滑动窗口的开始滑动的偏移； 默认 [0，0]
* grid_height : 高度方向滑动的次数
* grid_width  : 宽度方向滑动的次数

其中

1. scales 和 aspect_ratios 组合记录所有的缩放比例
2. grid_height * grid_width 记录了总共需要滑动的次数

整个实现就是将 base_anchor_size 的每一个子图，
以 anchor_stride 和 anchor_offset 的组合为步调，
滑动  grid_height * grid_width 次，返回所有所有的滑动窗口

### 例子

scales = [1.0, 0.5]
aspect_ratios = [1.0, 2.0]
base_anchor_size = [16, 16] # 滑动窗口大小为 16 x 16 的正方形

将 base_anchor_size  [1.0, 1.0], [1.0, 2.0], [0.5, 1.0], [0.5, 2.0]
的比例进行缩放

grid_height, grid_width = (2,2) # 高度和宽度方向都滑动 2 次
anchor_stride = [4, 3]  # 横轴每次滑动 4，纵轴每次滑动 3
anchor_offset = [1, 2]  # 横轴偏移为 1，纵轴偏移为 2

因此有组合 [1 2] [1 5] [5 2] [5 5]，因此每张图片总共要滑动四次，即 4 个滑动窗口


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

  y_centers = np.arange(grid_height)
  # 纵坐标要滑动的点
  y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
  x_centers = np.arange(grid_width)
  # 横坐标要滑动的点
  x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
  x_centers, y_centers = np.meshgrid(x_centers, y_centers)

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
  grid_height, grid_width = (2,2)
  anchor_stride = [4, 3]
  anchor_offset = [1, 2]
  scales = [1.0]
  aspect_ratios = [1.0]

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
