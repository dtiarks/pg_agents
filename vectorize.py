import numpy as np


def flatten_array_list(a_list):
	return np.concatenate([l.ravel() for l in a_list])

def unflatten_array_list(array,shapes):
	c=0
	l=[]
	for s in shapes:
		tmp=np.array(array[c:c+np.prod(s)])
		reshaped=np.reshape(tmp,s)
		c+=np.prod(s)

		l.append(reshaped)

	return l

# shape_list=[(2,2,1),(3,3),(4,4)]

# array_list=[s[0]*np.ones(s) for s in shape_list]

# flat=flatten_array_list(array_list)

# unflat=unflatten_array_list(flat,shape_list)

# print(array_list)
# print(unflat)
