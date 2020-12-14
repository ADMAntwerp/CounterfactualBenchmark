import tensorflow as tf

C = 1000.
D = 1000.


def op_gt(op1, op2, use_tensor, scale=100.):
    if use_tensor:
        return (C / scale) * tf.exp(-(D / scale) * (op1 - op2))
    else:
        return op1 > op2


def op_lt(op1, op2, use_tensor, scale=100.):
    if use_tensor:
        return (C / scale) * tf.exp((D / scale) * (op1 - op2))
    else:
        return op1 < op2


def op_and(op1, op2, use_tensor):
    return op1 + op2 if use_tensor else op1 and op2


def op_neq(op1, op2, use_tensor):
    return 10e5 * tf.exp(-50 * tf.square(op1 - op2)) if use_tensor else op1 != op2
