


def clip_by_norm(t, clip_norm, axes=None, name=None)

返回 t * clip_norm * tf.minimum(1/clip_norm,  rsqrt(tf.reduct_sum(t*t), axes))
