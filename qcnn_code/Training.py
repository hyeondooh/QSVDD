# Implementation of Quantum circuit training procedure

import QCNN_circuit
# import Hierarchical_circuit

import pennylane as qml
#import numpy as np
from pennylane import numpy as np

import autograd.numpy as anp
import torch
from torch.optim import Adam, Adagrad


def qae_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        # loss = loss + (l - p) ** 2
        # loss = loss + (l - np.sum(p))
        loss = loss + (-np.sum(p))
    loss = loss / len(labels)
    return loss

def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy

    return -1 * loss

def svdd_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels,predictions):
        # loss = loss + abs(p - l).mean()
        # loss = loss + np.sum(abs(p-l))
        loss = loss + np.sum((p-l)**2)
        
        # loss = loss + np.sqrt(np.sum(p-l)**2 + 1) -1
    loss = loss / len(labels)
    return loss




def cost(params, X, Y, U, U_params, embedding_type, circuit, cost_fn, measure_axis):
    if circuit == 'QCNN':
        predictions = [QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn=cost_fn, measure_axis=measure_axis) for x in X]
    # elif circuit == 'Hierarchical':
    #     predictions = [Hierarchical_circuit.Hierarchical_classifier(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x in X]

    if cost_fn == 'qae':
        loss = qae_loss(Y, predictions)
    elif cost_fn == 'cross_entropy':
        loss = cross_entropy(Y, predictions)
    elif cost_fn == 'svdd':
        loss = svdd_loss(Y, predictions)
    return loss

# Circuit training parameters

steps = 2000
learning_rate = 0.001
batch_size = 16


def circuit_training(X_train, Y_train, U, U_params, embedding_type, circuit, cost_fn, measure_axis):
    if (circuit == 'QCNN')&(U != 'U_SU4_no_pooling'):
        total_params = U_params * 3 + 2 * 3
    elif (circuit == 'QCNN')&(U == 'U_SU4_no_pooling'):
        # par : 45
        # total_params = U_params * 3

        # par : 45 -> 90
        # total_params = U_params * 6
        total_params = U_params * 7

        # par : 90 -> 180
        # total_params = U_params * 12
    elif circuit == 'Hierarchical':
        total_params = U_params * 7
        
        
    ## Initializing parameter
    init_params = np.random.randn(total_params, requires_grad = True)
    params = init_params
    # params = np.random.randn(total_params, requires_grad = True)
    # params = np.array([-6.40203670e-01, -2.14747623e+00, -5.45937766e-01,
    #      1.26944305e+00, -1.11386199e+00,  1.98537280e-01,
    #      2.42954253e-02,  4.73088420e-02,  3.30500201e-02,
    #     -8.17918300e-01, -1.23613796e-01, -2.18387797e-01,
    #     -1.81021691e+00, -1.12981004e-01, -2.77286909e+00,
    #      1.05619401e+00,  3.30314238e-01, -9.86462466e-01,
    #     -5.71194632e-01, -8.69921971e-01, -1.02187512e-01,
    #      5.77077839e-02,  1.52983346e+00,  1.06619996e-01,
    #      4.31374205e-01, -8.54841345e-01, -1.55794980e+00,
    #      6.89205529e-01,  2.27144732e-01,  7.83572085e-01,
    #      1.40447007e+00, -2.36780057e-01,  1.77891860e+00,
    #      1.31989117e+00, -7.65584795e-01, -1.26029592e-01,
    #     -1.47076842e+00, -1.55146013e+00, -3.55486177e+00,
    #     -8.48564736e-01, -1.24512018e+00,  3.01612577e-01,
    #      1.13643694e+00, -1.85928282e+00,  6.08906525e-01,
    #     -6.46758916e-01,  2.65536456e+00, -1.36201996e+00,
    #     -6.38842083e-01,  3.84250527e-01,  1.46377268e+00,
    #      2.79582817e-03, -3.08836956e-03, -3.15023892e+00,
    #      1.18533469e+00,  2.45576615e+00, -1.22236921e+00,
    #     -1.19164797e+00,  1.84247392e+00, -6.89866828e-01,
    #      4.32598980e-01, -4.86902696e-01, -7.84550987e-01,
    #      6.15642767e-01, -1.90495310e-01, -8.16988904e-01,
    #      1.85607375e+00, -7.27221010e-01,  4.27010106e-01,
    #      1.95375734e+00,  1.17426942e+00, -9.25397149e-01,
    #     -1.29035989e+00,  6.74361672e-01,  2.45622970e-01,
    #      1.16449092e+00,  5.23003764e-01, -9.19607169e-01,
    #     -2.52475577e-01,  5.96943977e-01,  6.38003809e-01,
    #     -2.76724138e-01, -8.28090521e-01,  1.48720623e+00,
    #     -2.02695980e+00,  1.08321493e+00, -1.88471985e-01,
    #      1.39585776e-01, -3.65390780e-01, -1.18355703e+00,
    #      5.85235529e-01,  1.67327718e+00, -6.70681377e-01,
    #      1.25302960e+00,  1.63613824e+00, -7.92552659e-02,
    #     -1.44329315e-01,  1.16891675e+00,  8.57254122e-01,
    #     -4.43522517e-01,  6.72765509e-01,  1.07703765e+00,
    #      2.48868685e+00, -6.85405494e-01, -1.79708947e+00],requires_grad=True)



    # Optimizer method
    opt = qml.AdamOptimizer(stepsize=learning_rate)    
#    opt = qml.NesterovMomentumOptimizer(stepsize = learning_rate, momentum=0.9)
#    param_history = [params]
    param_history= [params]
    loss_history = []

#    params = torch.tensor(np.random.randn(total_params),requires_grad=True).detach().numpy()
#    params = np.random.randn(total_params, requires_grad=True)   
#    params = torch.rand(total_params, requires_grad=True)


    for it in range(steps):

        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]

        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn, measure_axis), params)

        param_history.append(params)        
        loss_history.append(cost_new)

        if it % 5 == 0:
            print("iteration: ", it, " cost: ", cost_new)


    return loss_history, params, param_history


