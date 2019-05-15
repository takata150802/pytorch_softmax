#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 20:49:37 2019

@author: ryotakata
"""
import torch
import torchvision
import numpy as np
torch.manual_seed(0)

def main ():
    version([torch, torchvision, np])
    
    ABS_ERROR = 10e-5
    RAND_SCALE = 100
    MAX_LEN = 20 # maximun value of (M, C, H, W)
    
    for _ in range (100):
        """
        size = (N, C, H, W)
        s.t.
        {x, y, dy, dx}.shape == size
        x, y, dy and dx are input/output tensor of softmax_forward/backward 
        {N, C, H, W} are randint(1, MAX_LEN)
        """
        size = torch.randint(low=1,high=MAX_LEN,size=(4,))
        size = tuple(deepcopy_torch_tensor(size).numpy())
        x = torch.rand(*size) * RAND_SCALE
        dy = torch.rand(*size) * RAND_SCALE
        test_pytorch_vs_pseudo_softmax_backward(x, dy, ABS_ERROR)
    return 

def version(ll=[torch, torchvision, np]):
    import platform
    print ("Python ", end='')
    print (platform.python_version(), end='')
    print ()
    for i in ll:
        print(i.__name__, end='')
        print("==", end='')
        print(i.__version__, end='')
        print()
    return

def test_pytorch_vs_pseudo_softmax_backward (x, dy, abs_error = 10e-5):
    """
    return abs(expected - actual) <= abs_err
    s.t.
    expected: dx calculated by torch.softmax(x).backward(dy)
    actual: dx, numpy implementation of the pseudo code
    """
    assert x.shape == dy.shape


    """forward"""
    x = x.requires_grad_(True)
    y = torch.softmax(x, dim = 1)
    
    """backward"""
    exp_dx = pytorch_softmax_backward (x, y, dy)
    act_dx = pseudo_softmax_backward (y, dy)
    
    err = exp_dx - act_dx
    max_err = abs(err).max()
    
    msg = "{x, y, dy, dx}.shape == " + str(x.shape)
    if max_err <= abs_error:
        print ("[OK] " + msg)
        return True
    else:
        print ("[NG] " + msg)
        return False

def pytorch_softmax_backward (x, y, dy):
    """
    return dx calculated by torch.softmax(x).backward(dy) 
    {x, y, dy, dx} are torch.Tensor
    {x, y, dy, dx}.dim() == 4
    {x,y}.requires_grad == True
    y.grad_fn != None
    """    
    assert is_4d_torch_tensor(y)
    assert is_4d_torch_tensor(dy)
    assert x.requires_grad
    assert y.requires_grad and y.grad_fn != None
    
    """
    if torch.Tensor.requires_grad == True:
         torch.Tensor.grad to be calculated.
    """
    y.backward(gradient = dy)
    
    dx = x.grad
    assert is_4d_torch_tensor(dx)
    return dx

def pseudo_softmax_backward (tt_y, tt_dy):
    """
    numpy implementation of SoftmaxBackward pseudo code
    mode = SOFTMAX_MODE_CHANNEL
    {tt_y, tt_dy, tt_dx} are torch.Tensor
    {tt_y, tt_dy, tt_dx}.dim() == 4
    """
    assert is_4d_torch_tensor(tt_y)
    assert is_4d_torch_tensor(tt_dy)
    
    """torch.Tensor -> np.ndarry"""
    y = deepcopy_torch_tensor(tt_y).numpy().astype(np.float32)
    dy = deepcopy_torch_tensor(tt_dy).numpy().astype(np.float32)
    
    N, C, H, W = y.shape
    np_dx = np.zeros(shape=(N, C, H, W)).astype(np.float32) * np.nan
    sum_ =  np.zeros(shape=(N, 1, H, W)).astype(np.float32)
    
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    sum_[n,0,h,w] += dy[n,c,h,w] * y[n,c,h,w]
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    np_dx[n,c,h,w] = y[n,c,h,w] * (dy[n,c,h,w] - sum_[n,0,h,w])

    tt_dx = torch.from_numpy(np_dx)
    assert is_4d_torch_tensor(tt_dx)
    return tt_dx
    
def is_4d_torch_tensor(tt):
    return isinstance(tt, torch.Tensor) and tt.dim() == 4

def deepcopy_torch_tensor(tt):
    ret = tt.copy_(tt)
    ret = ret.detach()
    return ret

if __name__ == '__main__':
    main()
