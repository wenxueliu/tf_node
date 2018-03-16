
 问题

    static_shape = image.get_shape().with_rank(rank).as_list()
    dynamic_shape = array_ops.unstack(array_ops.shape(image), rank)

static_shape 和  dynamic_shape 的关系是什么，有什么区别



## Image 处理

tensorflow/python/ops/image_ops.py

对于读到的图片内容，先解码，之后开始处理，如翻转， 调整对比，调整亮度等等

def decode_image(contents, channels=None, name=None) : 将图片解码为 Tensor

def random_flip_left_right(image, seed=None)

def adjust_contrast(images, contrast_factor)

def adjust_brightness(image, delta) : 调整亮度

def crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)

截取  image 的高度 offset_height, target_height, 宽度 [offset_width, target_width]

def pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width)

image 的 height，前面 offset_height 部分补0，后面  target_height - offset_height - image.height 部分补 0
image 的 width，前面 offset_width 部分补0，后面  target_width - offset_width - image.width 部分补 0

def resize_images(images, size, method=ResizeMethod.BILINEAR, align_corners=False)

根据 method 将 images 变为 size 大小的 images

* `ResizeMethod.BILINEAR`</b>: [Bilinear interpolation.]( https://en.wikipedia.org/wiki/Bilinear_interpolation)
* `ResizeMethod.NEAREST_NEIGHBOR`</b>: [Nearest neighbor interpolation.]( https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
* `ResizeMethod.BICUBIC`</b>: [Bicubic interpolation.]( https://en.wikipedia.org/wiki/Bicubic_interpolation)
* `ResizeMethod.AREA`</b>: Area interpolation.

def rgb_to_grayscale(images, name=None)

把 images 的每个 channel 对应的值加起来乘以[0.2989, 0.5870, 0.1140] (此时 shape 为 [height, width])，之后转换为 [height, width, 1]

参考 https://en.wikipedia.org/wiki/Luma_%28video%29

def grayscale_to_rgb(images, name=None)
