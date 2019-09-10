import inspect, torch, typing


def convert_to_tensors_wrapper(f):
    """A wrapper that converts all declared tensor types to tensor"""
    def decorate(*args, **kwargs):
        sig = inspect.signature(f)
        args = list(args)
        for i, var in enumerate(sig.parameters.values()):
            if var.annotation == torch.Tensor:
                args[i] = torch.Tensor(args[i]) # try converting to tensor
                assert len(args[i].shape) == 3

        return f(*args, **kwargs)

    return decorate


def batchify_tensors(f):
    """A wrapper that converts all declared tensor types to tensor"""
    def decorate(*args, **kwargs):
        sig = inspect.signature(f)
        args = list(args)
        for i, var in enumerate(sig.parameters.values()):
            if var.annotation == torch.Tensor:
                args[i] = torch.Tensor(args[i])  # try converting to tensor
                assert len(args[i].shape) == 3

        return f(*args, **kwargs)

    return decorate



@convert_to_tensors_wrapper
def func(a: int, b: str, c: torch.Tensor):
    print(c)

# f = test_wrapper(func)
func(1,2,[[[1,2], [3,4]]])
#sig = inspect.signature(func)
#for var in sig.parameters.values():
#    print(var.annotation == type(torch.Tensor()))
#    print(var.annotation == torch.Tensor)
#print(torch.Tensor in inspect.signature(func).values())
