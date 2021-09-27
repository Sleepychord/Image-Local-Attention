import torch
from torch.nn import functional as F
from function import f_similar, TorchLocalAttention
import time


def check(a, b):
    tmp = torch.max(torch.stack((a.abs(), b.abs())), dim=0)[0]
    idx = tmp > 0
    # return ( (a-b).abs()[idx] / tmp[idx]) .max()
    return (a-b).abs().max()

def test_correct2(h, w, c, kh, kw, casual_mask=False):
    x1 = torch.rand(32, c, h, w).cuda()
    y1 = torch.rand(8, c, h, w).cuda()
    x2 = x1.clone()#.half()
    y2 = y1.clone()#.half()
    x3 = x1.clone()#.half()
    y3 = y1.clone()#.half()
    
    x1.requires_grad_()
    y1.requires_grad_()
    x2.requires_grad_()
    y2.requires_grad_()
    x3.requires_grad_()
    y3.requires_grad_()

    y1t = y1.view(y1.shape[0], 1, c, h, w).expand(y1.shape[0], 4, c, h, w).reshape(-1, c, h, w)

    z1 = TorchLocalAttention.f_similar(x1, y1t, kh, kw, casual_mask)
    z2 = f_similar(x2, y2, kh, kw, casual_mask)
    z2 = z2 * (z2>-1)
    x3t = x3.view(8, 2, 2, c, h, w).permute(0, 3, 4, 1, 5, 2).reshape(8, c, h * 2, w * 2)
    z3 = f_similar(x3t, y3, kh, kw, casual_mask).view(8, h, 2, w, 2, -1).permute(0, 2, 4, 1, 3, 5).reshape(32, h, w, -1)
    z3 = z3 * (z3>-1)

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
    print("maximum difference 2: {:.5f}\t{:.5f}\t{:.5f}".format(err1.item(), err2.item(), err3.item()))


    
def test_correct(h, w, c, kh, kw, casual_mask=False):
    x1 = torch.rand(40, c, h, w).cuda()
    y1 = torch.rand(40, c, h, w).cuda()
    x2 = x1.clone()#.half()
    y2 = y1.clone()#.half()
    x1.requires_grad_()
    y1.requires_grad_()
    x2.requires_grad_()
    y2.requires_grad_()
    z1 = TorchLocalAttention.f_similar(x1, y1, kh, kw, casual_mask)
    z2 = f_similar(x2, y2, kh, kw, casual_mask)
    z2 = z2 * (z2>-1000)
    grad = torch.ones(z1.size()).cuda()

    z1.sum().backward()
    z2.sum().backward()
    # z1.backward(grad)
    # z2.backward(grad)
    err1 = check(z1.data, z2.data)
    err2 = check(x1.grad.data, x2.grad.data)
    err3 = check(y1.grad.data, y2.grad.data)
    print("maximum difference: {:.5f}\t{:.5f}\t{:.5f}".format(err1.item(), err2.item(), err3.item()))

    
def test_efficiency_forward(h, w, c, kh, kw, casual_mask=False):
    x = torch.rand(80, c, h, w).cuda().half()
    y = torch.rand(80, c, h, w).cuda().half()

    with torch.no_grad():
        torch.cuda.reset_max_memory_allocated()
        z = f_similar(x, y, kh, kw, casual_mask)
        memory = torch.cuda.max_memory_allocated() / 1000000
        del z

    with torch.no_grad():
        torch.cuda.reset_max_memory_allocated()
        z = TorchLocalAttention.f_similar(x, y, kh, kw, casual_mask)
        memory_torch = torch.cuda.max_memory_allocated() / 1000000
        del z
        torch.cuda.empty_cache()

    with torch.no_grad():
        torch.cuda.synchronize()
        t = time.time()
        for i in range(20):
            z = f_similar(x, y, kh, kw, casual_mask)
        torch.cuda.synchronize()
        t = (time.time() - t) / 20
        del z
        
        torch.cuda.synchronize()
        t_torch = time.time()
        for i in range(3):
            z = TorchLocalAttention.f_similar(x, y, kh, kw, casual_mask)
        torch.cuda.synchronize()
        t_torch = (time.time() - t_torch) / 3
        del z
    print("{:.2f},{:.2f}||{:.5f},{:.5f}".format(memory_torch, memory, t_torch, t))
    # print("{:.2f},{:.2f}||{:.5f},{:.5f}".format(memory, memory, t, t))

    
def test_efficiency_backward(h, w, c, kh, kw, casual_mask=False):

    x = torch.rand(20, c, h, w).cuda().half()
    y = torch.rand(20, c, h, w).cuda().half()
    x.requires_grad_()
    y.requires_grad_()

    torch.cuda.reset_max_memory_allocated()
    z = f_similar(x, y, kh, kw, casual_mask)
    grad = torch.rand(z.size()).cuda()
    z.backward(grad)
    memory = torch.cuda.max_memory_allocated() / 1000000
    x.grad.data.zero_()
    y.grad.data.zero_()
    del z

    torch.cuda.reset_max_memory_allocated()
    z = TorchLocalAttention.f_similar(x, y, kh, kw, casual_mask)
    grad = torch.rand(z.size()).cuda()
    z.backward(grad)
    memory_torch = torch.cuda.max_memory_allocated() / 1000000
    x.grad.data.zero_()
    y.grad.data.zero_()
    del z

    torch.cuda.synchronize()
    t = time.time()
    for i in range(3):
        z = f_similar(x, y, kh, kw, casual_mask)
        z.backward(grad)
        x.grad.data.zero_()
        y.grad.data.zero_()
    torch.cuda.synchronize()
    t = (time.time() - t) / 3
    del z

    torch.cuda.synchronize()
    t_torch = time.time()
    for i in range(3):
        z = TorchLocalAttention.f_similar(x, y, kh, kw, casual_mask)
        z.backward(grad)
        x.grad.data.zero_()
        y.grad.data.zero_()
    torch.cuda.synchronize()
    t_torch = (time.time() - t_torch) / 3
    del z
    print("{:.2f},{:.2f}||{:.5f},{:.5f}".format(memory_torch, memory, t_torch, t))
    # print("{:.2f},{:.2f}||{:.5f},{:.5f}".format(memory, memory, t, t))


if __name__ == '__main__':
    # torch.backends.cuda.matmul.allow_tf32 = False
    for im in [64]:
        for c in [64]:
            for block in [7]:
                print("input:{} channel:{} block:{}".format(im, c, block))
                # test_correct2(im, im, c, block, block)
                test_correct(im, im, c, block, block, False)
                test_efficiency_forward(im, im, c, block, block)
                # test_efficiency_forward(im, im, c, block, block, True)
                test_efficiency_backward(im, im, c, block, block)
                # test_efficiency_backward(im, im, c, block, block, True)
    
