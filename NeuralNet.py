# Objective: create a neural network with two layers
import numpy as np

def  nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)

        return 1/(1+np.exp(-x))


# input data
x = np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])

# output data
y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

# create the synapses
syn0 = 2*np.random.random((3,4)) - 1    # initialising using random weights
syn1 = 2*np.random.random((4,1)) - 1

# print(syn0)

# training step
for j in range(10000000):

    l0 = x
    l1 = nonlin(np.dot(l0, syn0), deriv=False)  # np.dot is the matrix product
    l2 = nonlin(np.dot(l1, syn1), deriv=False)

    l2_error = y - l2              # calculate the error by comparing to the output vector

    if(j % 100000) == 0:             # % is the modulus operator. If the remainder is 0 then print update
        print("Error", str( np.mean( np.abs( l2_error ) ) ))

        l2_delta = l2_error*nonlin(l2, deriv=True)

        l1_error = l2_delta.dot(syn1.T)

        l1_delta = l1_error * nonlin(l1,deriv=True)

        # update weights
        syn1 += l1.T.dot(l2_delta)   # += adds the delta to to synapse 1
        syn0 += l0.T.dot(l1_delta)

print('Output after training')
print(l2)               # The vector is supposed to be [0,1,1,0]. The results are usually not close
                        # implying that this neural net is not very good.
