
待计算的 Tensor 允许的类型 [dtypes.float16, dtypes.float32, dtypes.float64]

def compute_gradients(self, loss, var_list=None,
                 gate_gradients=GATE_OP,
                 aggregation_method=None,
                 colocate_gradients_with_ops=False,
                 grad_loss=None):

var_list :  如果为 None，则来自 ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES 和 ops.GraphKeys._STREAMING_MODEL_PORTS


