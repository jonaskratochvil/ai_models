from random import randint
import torch


class TensorQueue:

    def __init__(self, max_size : int):
        """

        If `queue` reaches max size, every push will pop the oldest items

        :param max_size:
        """
        self.queue = torch.Tensor([])
        self.max_size = max_size

    def __len__(self):
        return len(self.queue)

    def push(self,x: torch.Tensor) -> torch.Tensor:
        """Push one or more items

        If Queue is full, the oldest items are poped.

        :param x:
        :return:
        """

        if len(x) + len(self) > self.max_size:
            to_pop = self.pop(len(x) + len(self) - self.max_size)
        else:
            to_pop = torch.Tensor([])

        self.queue = torch.cat((self.queue,
                                x[-self.max_size:]))
        return to_pop

    def pop(self, num: int = 1) -> torch.Tensor:
        """Pop `num` items

        :param num: How many items to pop. If `num` >= `self.max_size`, everything is poped
        :return: list
        """
        to_pop = self.queue[:num]
        self.queue = self.queue[num:]
        return to_pop


def random_chunks(lst: list, min_chunk_size: int, max_chunk_size: int) -> list:

    chunks, i, j = [], 0, 0

    while i <= len(lst) - max_chunk_size:
        j = randint(min_chunk_size, min(max_chunk_size, len(lst)-min_chunk_size - i))
        chunks.append(lst[i:i+j])
        i += j

    chunks.append(lst[i:])
    return chunks
