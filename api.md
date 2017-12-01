
tf.reshape
    reshape(
        tensor,
        shape,
        name=None
    )

    将 tensor 转变为 shape 指定的维度

tf.argmax : tensor 中值最大的索引

    x = [1, 3, 4, 5, 3, 2] 返回 3

    x = [[1, 3, 4], [5, 3, 2]] 返回 [1, 0, 0]

tf.cast : 进行类型诊断

    tf.constant([1.8, 2.2], dtype=tf.float)
    tf.cast(a, tf.int32) ==> [1, 2]

tf.summary.scalar

    将 summary 协议转换为 single scalar value
