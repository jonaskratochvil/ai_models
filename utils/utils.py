from random import randint


def random_chunks(lst: list, min_chunk_size: int, max_chunk_size: int) -> list:

    chunks, i, j = [], 0, 0

    while i <= len(lst) - max_chunk_size:
        j = randint(min_chunk_size, min(max_chunk_size, len(lst)-min_chunk_size - i))
        chunks.append(lst[i:i+j])
        i += j

    chunks.append(lst[i:])
    return chunks

