import tensorflow as tf


def mvn_kl_div_full(a, b, dim):
    co0 = a.co
    co1 = b.co
    co1_i = tf.matrix_inverse(co1)
    mu = b.means - a.means
    k = dim

    A = tf.trace(tf.matmul(co0, co1_i))
    B = tf.squeeze(tf.matmul(tf.matmul(tf.expand_dims(mu, 1), co1_i), tf.expand_dims(mu, 2)), axis=[1, 2])

    det_co1 = tf.matrix_determinant(co1)
    det_co0 = tf.matrix_determinant(co0)
    co_rat = tf.clip_by_value(det_co1 / det_co0, 1e-12, 1e12)
    C = tf.log(co_rat)
    ret = 0.5 * (A + B - k + C)
    return ret


def mvn_kl_div_diag(a, b, dim):
    co0 = a.co
    co1 = b.co
    co0_diag = tf.matrix_diag_part(co0)
    co1_diag = tf.matrix_diag_part(co1)
    co1_i = tf.matrix_diag(1. / co1_diag)

    det_co1 = tf.reduce_prod(co1_diag, axis=1)
    det_co0 = tf.reduce_prod(co0_diag, axis=1)

    mu = b.means - a.means
    k = dim

    A = tf.trace(tf.matmul(co0, co1_i))
    B = tf.squeeze(tf.matmul(tf.matmul(tf.expand_dims(mu, 1), co1_i), tf.expand_dims(mu, 2)), axis=[1, 2])

    co_rat = tf.clip_by_value(det_co1 / det_co0, 1e-12, 1e12)
    C = tf.log(co_rat)
    ret = 0.5 * (A + B - k + C)
    return ret


def uvn_kl_div(a, b):
    s0 = a.std
    s1 = b.std

    s_rat = tf.clip_by_value(s1/s0, 1e-12, 1e12)

    mu = a.means - b.means

    A = tf.log(s_rat)
    B = (s0**2+mu**2)/(2*s1**2+1e-12)
    C = -0.5
    ret = A + B + C
    return ret


def kl_div(a, b, dim):
    if dim==1:
        return uvn_kl_div(a, b)
    else:
        return mvn_kl_div_diag(a, b, dim)