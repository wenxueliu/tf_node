
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None)

1. input 的维度与 depth 一致并且 stride = 1

relu(input + (conv2d(1x1, 1, depth_bottleneck) + conv2d(3x3, 1, depth_bottleneck) + conv2d(1x1, 1, depth)))

2. input 的维度与 depth 一致并且 stride > 1

relu(max_pool2d(1x1, stride) + (conv2d(1x1, depth_bottleneck, 1) + conv2d_vaild(3x3, stride, depth_bottleneck) + conv2d(1x1, 1, depth))

3. input 的维度与 depth 不一致并且 stride = 1

relu(conv(1x1, 1, depth) + (conv2d(1x1, depth_bottleneck, 1) + conv2d(3x3, 1, depth_bottleneck) + conv2d(1x1, 1, depth))

4. input 的维度与 depth 不一致并且 stride > 1

relu(conv(1x1, stride, depth) + (conv2d(1x1, depth_bottleneck, 1) + conv2d_vaild(3x3, stride, depth_bottleneck) + conv2d(1x1, 1, depth))

问题：

1. pading 的计算
2. 什么是 dense prediction tasks ? such as semantic segmentation or object detection
3. atrous convolution
