import numpy as np


def flatten_array_list(a_list):
    return np.concatenate([l.ravel() for l in a_list])


def unflatten_array_list(array, shapes):
    c = 0
    l = []
    for s in shapes:
        tmp = np.array(array[c:c + np.prod(s)])
        reshaped = np.reshape(tmp, s)
        c += np.prod(s)

        l.append(reshaped)

    return l


def cg_solve(Ax_prod, b, x_init):
    x = x_init

    r = b - Ax_prod(x)
    p = r

    rsold = np.dot(r, r)

    for i in range(20):
        Ap = Ax_prod(p)

        alpha = rsold / np.dot(p, Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = np.dot(r, r)
        # print(np.sqrt(rsnew))
        if (np.sqrt(rsnew) < 1e-10):
            return x

        beta = np.true_divide(rsnew, rsold)

        p = r + beta * p
        rsold = rsnew

    return x

# A=[[4.,1.],[1.,3.]]
# b=[1.,2.]
# x_init=[2.,1.]

# Ax_prod=lambda x:np.dot(A,x)

# print(cg_solve(Ax_prod,b,x_init))

# shape_list=[(2,2,1),(3,3),(4,4)]

# array_list=[s[0]*np.ones(s) for s in shape_list]

# flat=flatten_array_list(array_list)

# unflat=unflatten_array_list(flat,shape_list)

# print(array_list)
# print(unflat)
