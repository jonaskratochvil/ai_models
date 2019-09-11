import pytest, os, sys

# import project root to PYTHONPATH
abs_dir = os.path.dirname(os.path.abspath(__file__)) # abs adress of this file
root_folder = abs_dir.split('tests')[0]
sys.path.append(root_folder)

from utils.utils import random_chunks, TensorQueue


def test_random_chunks():
    lower, upper = 3, 7
    lst = [i for i in range(23)]
    for i in range(10):
        for chunk in random_chunks(lst, lower, upper):
            assert lower <= len(chunk) <= upper


def test_tensor_queue():
    import torch
    queue = TensorQueue(5)
    a = torch.Tensor([1,2,3])
    b = torch.Tensor([4, 5, 6, 7, 8, 9])
    queue.push(a)
    assert torch.all(torch.eq(queue.q, a))
    queue.push(b[:1])
    assert torch.all(torch.eq(queue.q, torch.Tensor([1, 2, 3, 4])))
    queue.push(b)
    assert torch.all(torch.eq(queue.q, torch.Tensor([5, 6, 7, 8, 9])))
    x = queue.push(a)
    assert torch.all(torch.eq(queue.q, torch.Tensor([8, 9, 1, 2, 3])))
    assert torch.all(torch.eq(x, torch.Tensor([5, 6, 7])))
    queue.pop(2)
    assert torch.all(torch.eq(queue.q, a))

    # check if the last dim is concateated
    queue = TensorQueue(5)
    a = torch.Tensor([[[1,2,3]]]).permute(0,2,1)
    b = torch.Tensor([[[4, 5, 6]]]).permute(0,2,1)
    queue.push(a)
    assert torch.all(torch.eq(queue.q, a))
    queue.push(b)
    assert torch.all(torch.eq(queue.q, torch.Tensor([[[1, 2, 3], [4, 5, 6]]]).permute(0, 2, 1)))
