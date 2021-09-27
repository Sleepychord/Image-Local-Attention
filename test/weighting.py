import torch
from torch.nn import functional as F
from function import f_weighting, TorchLocalAttention
import time


def check(a, b):
    tmp = torch.max(torch.stack((a.abs(), b.abs())), dim=0)[0]
    idx = tmp > 0
    # return ( (a-b).abs()[idx] / tmp[idx]) .max()
    return (a-b).abs().max()

def test_correct2(h, w, c, kh, kw, casual_mask=False):
    patch = kh*kw // 2 +1 if casual_mask else kh*kw 
    x1 = torch.rand(4, c, h, w).cuda()
    y1 = torch.rand(16, h, w, patch).cuda()
    x2 = x1.clone()
    y2 = y1.clone()
    x3 = x1.clone()
    y3 = y1.clone()

    x1.requires_grad_()
    y1.requires_grad_()
    x2.requires_grad_()
    y2.requires_grad_()
    x3.requires_grad_()
    y3.requires_grad_()

    x1t = x1.view(x1.shape[0], 1, c, h, w).expand(x1.shape[0], 4, c, h, w).reshape(-1, c, h, w)

    z1 = TorchLocalAttention.f_weighting(x1t, y1, kh, kw, casual_mask)
    z2 = f_weighting(x2, y2, kh, kw, casual_mask)

    y3t = y3.view(y1.shape[0]//4, 2, 2, h, w, patch).permute(0, 3, 1, 4, 2, 5).reshape(y1.shape[0]//4, h*2, w*2, patch) #
    z3 = f_weighting(x3, y3t, kh, kw, casual_mask).view(y1.shape[0]//4, c, h, 2, w, 2).permute(0, 3, 5, 1, 2, 4).reshape(y1.shape[0], c, h, w)

    grad = torch.rand(z1.size()).cuda()

    z1.backward(grad)
    z2.backward(grad)
    z3.backward(grad)

    err1 = check(z1.data, z2.data)
    err2 = check(x1.grad.data, x2.grad.data)
    err3 = check(y1.grad.data, y2.grad.data)
    print("maximum difference: {:.5f}\t{:.5f}\t{:.5f}".format(err1.item(), err2.item(), err3.item()))
    err1 = check(z1.data, z3.data)
    err2 = check(x1.grad.data, x3.grad.data)
    err3 = check(y1.grad.data, y3.grad.data)
    print("maximum difference: {:.5f}\t{:.5f}\t{:.5f}".format(err1.item(), err2.item(), err3.item()))


def test_correct(h, w, c, kh, kw, casual_mask=False):
    patch = kh*kw // 2 +1 if casual_mask else kh*kw 
    x1 = torch.rand(4, c, h, w).cuda() 
    y1 = torch.rand(4, h, w, patch).cuda() 
    x2 = x1.clone()
    y2 = y1.clone()

    x1.requires_grad_()
    y1.requires_grad_()
    x2.requires_grad_()
    y2.requires_grad_()

    z1 = TorchLocalAttention.f_weighting(x1, y1, kh, kw, casual_mask)
    z2 = f_weighting(x2, y2, kh, kw, casual_mask)
    grad = torch.rand(z1.size()).cuda()

    z1.backward(grad)
    z2.backward(grad)

    err1 = check(z1.data, z2.data)
    err2 = check(x1.grad.data, x2.grad.data)
    err3 = check(y1.grad.data, y2.grad.data)
    print("maximum difference: {:.5f}\t{:.5f}\t{:.5f}".format(err1.item(), err2.item(), err3.item()))

    
def test_efficiency_forward(h, w, c, kh, kw, casual_mask=False):
    patch = kh*kw // 2 +1 if casual_mask else kh*kw 
    x = torch.rand(20, c, h, w).cuda().half()
    y = torch.rand(20, h, w, patch).cuda().half()

    with torch.no_grad():
        torch.cuda.reset_max_memory_allocated()
        z = f_weighting(x, y, kh, kw, casual_mask)
        memory = torch.cuda.max_memory_allocated() / 1000000
        del z

    with torch.no_grad():
        torch.cuda.reset_max_memory_allocated()
        z = TorchLocalAttention.f_weighting(x, y, kh, kw, casual_mask)
        memory_torch = torch.cuda.max_memory_allocated() / 1000000
        del z

    with torch.no_grad():
        torch.cuda.synchronize()
        t = time.time()
        for i in range(3):
            z = f_weighting(x, y, kh, kw, casual_mask)
        torch.cuda.synchronize()
        t = (time.time() - t) / 3
        del z

        torch.cuda.synchronize()
        t_torch = time.time()
        for i in range(3):
            z = TorchLocalAttention.f_weighting(x, y, kh, kw, casual_mask)
        torch.cuda.synchronize()
        t_torch = (time.time() - t_torch) / 3
        del z
    print("{:.2f},{:.2f}||{:.5f},{:.5f}".format(memory_torch, memory, t_torch, t))


def test_efficiency_backward(h, w, c, kh, kw, casual_mask=False):
    patch = kh*kw // 2 +1 if casual_mask else kh*kw 
    x = torch.rand(20, c, h, w).cuda().half()
    y = torch.rand(20, h, w, patch).cuda().half()
    x.requires_grad_()
    y.requires_grad_()

    torch.cuda.reset_max_memory_allocated()
    z = f_weighting(x, y, kh, kw, casual_mask)
    grad = torch.rand(z.size()).cuda()
    z.backward(grad)
    memory = torch.cuda.max_memory_allocated() / 1000000
    x.grad.data.zero_()
    y.grad.data.zero_()
    del z

    torch.cuda.reset_max_memory_allocated()
    z = TorchLocalAttention.f_weighting(x, y, kh, kw, casual_mask)
    grad = torch.rand(z.size()).cuda()
    z.backward(grad)
    memory_torch = torch.cuda.max_memory_allocated() / 1000000
    x.grad.data.zero_()
    y.grad.data.zero_()
    del z

    torch.cuda.synchronize()
    t = time.time()
    for i in range(3):
        z = f_weighting(x, y, kh, kw, casual_mask)
        z.backward(grad)
        x.grad.data.zero_()
        y.grad.data.zero_()
    torch.cuda.synchronize()
    t = (time.time() - t) / 3
    del z

    torch.cuda.synchronize()
    t_torch = time.time()
    for i in range(3):
        z = TorchLocalAttention.f_weighting(x, y, kh, kw, casual_mask)
        z.backward(grad)
        x.grad.data.zero_()
        y.grad.data.zero_()
    torch.cuda.synchronize()
    t_torch = (time.time() - t_torch) / 3
    del z
    print("{:.2f},{:.2f}||{:.5f},{:.5f}".format(memory_torch, memory, t_torch, t))


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    for im in [32]:
        for c in [32]:
            for block in [9]:
                print("input:{} channel:{} block:{}".format(im, c, block))
                test_correct(im, im, c, block, block, False)
                # test_correct2(im, im, c, block, block, True)
                # test_efficiency_forward(im, im, c, block, block)
                test_efficiency_forward(im, im, c, block, block, True)
                test_efficiency_backward(im, im, c, block, block)
                # test_efficiency_backward(im, im, c, block, block, True)
