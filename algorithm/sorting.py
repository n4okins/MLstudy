def bubble_sort(array: list, reverse: bool = False):
    array = array.copy()
    is_end = False
    end_index = 1
    length = len(array)

    while not is_end:
        is_swap = False
        for i in range(length - end_index):
            if not reverse and array[i] < array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                is_swap = True
            elif reverse and array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                is_swap = True

            yield array

        if not is_swap:
            is_end = True
    yield array


def shaker_sort(array: list, reverse: bool = False):
    array = array.copy()
    top_idx, btm_idx = 0, len(array) - 1
    while top_idx < btm_idx:
        last_swap_idx = top_idx
        for i in range(top_idx, btm_idx):
            if not reverse and array[i] < array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                last_swap_idx = i
            elif reverse and array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                last_swap_idx = i

            yield array
        btm_idx = last_swap_idx

        last_swap_idx = btm_idx
        for i in range(btm_idx, top_idx, -1):
            if not reverse and array[i - 1] < array[i]:
                array[i - 1], array[i] = array[i], array[i - 1]
                last_swap_idx = i
            elif reverse and array[i - 1] > array[i]:
                array[i - 1], array[i] = array[i], array[i - 1]
                last_swap_idx = i

            yield array

        top_idx = last_swap_idx
    yield array


def comb_sort(array: list, reverse: bool = False, shrink_factor: float = 1.3):
    array = array.copy()
    length = len(array)
    h = int(length / shrink_factor)
    is_swapped = True
    while h != 1 or is_swapped:
        is_swapped = False
        for i in range(0, length - h):
            if not reverse and array[i] < array[i + h]:
                array[i], array[i + h] = array[i + h], array[i]
                is_swapped = True
            elif reverse and array[i] > array[i + h]:
                array[i], array[i + h] = array[i + h], array[i]
                is_swapped = True
            yield array
        h = int(h / shrink_factor)
        if h < 1:
            break
    yield array


def gnome_sort(array: list, reverse: bool = False):
    array = array.copy()
    index = 0
    length = len(array)
    while index < length - 1:
        if index == -1:
            index = 0
        if not reverse and array[index] >= array[index + 1]:
            index += 1
        elif reverse and array[index] <= array[index + 1]:
            index += 1
        else:
            array[index], array[index + 1] = array[index + 1], array[index]
            index -= 1
        yield array


def merge_sort(array: list, reverse: bool = False):
    array = array.copy()

    def merge(start: int, end: int):
        if start < end - 1:
            middle = (start + end) // 2
            yield from merge(start, middle)
            yield from merge(middle, end)

            left = array[start:middle]
            right = array[middle:end]

            li, ri, ai = 0, 0, start
            while li < len(left) and ri < len(right):
                if reverse and left[li] < right[ri]:
                    array[ai] = left[li]
                    li += 1
                elif not reverse and left[li] > right[ri]:
                    array[ai] = left[li]
                    li += 1
                else:
                    array[ai] = right[ri]
                    ri += 1
                ai += 1

            while li < len(left):
                array[ai] = left[li]
                ai += 1
                li += 1

            while ri < len(right):
                array[ai] = right[ri]
                ai += 1
                ri += 1
            yield array

    yield from merge(0, len(array))


def select_sort(array: list, reverse: bool = False):
    array = array.copy()
    length = len(array)
    for i in range(length - 1):
        min_or_max = array[i]
        min_or_max_index = i
        for j in range(i + 1, length):
            if reverse and array[j] < min_or_max:
                min_or_max = array[j]
                min_or_max_index = j

            if not reverse and array[j] > min_or_max:
                min_or_max = array[j]
                min_or_max_index = j

        array[i], array[min_or_max_index] = array[min_or_max_index], array[i]
        yield array


def insertion_sort(array: list, reverse: bool = False):
    array = array.copy()
    for i in range(1, len(array)):
        array_i = array[i]
        for j in range(i - 1, -1, -1):
            if not reverse and array_i < array[j]:
                array[j + 1] = array_i
                break
            elif reverse and array_i >= array[j]:
                array[j + 1] = array_i
                break
            else:
                array[j], array[j + 1] = array[j + 1], array[j]
            yield array
        else:
            array[0] = array_i
        yield array


def shell_sort(array: list, reverse: bool = False):
    array = array.copy()
    length = len(array)
    h = length // 2
    while h > 0:
        for i in range(h, length):
            array_i = array[i]
            j = i - h
            while not reverse and j >= 0 and array_i < array[j]:
                array[j + h] = array[j]
                j -= h
            while reverse and j >= 0 and array_i > array[j]:
                array[j + h] = array[j]
                j -= h

            array[j + h] = array_i
            yield array
        h = h // 2


def quick_sort(array: list, reverse: bool = False):
    array = array.copy()
    length = len(array)
    if length < 2:
        yield array

    else:
        pivot_index = length // 2
        pivot = array[pivot_index]

        def divide(arr: list):
            ll, rr = [], []
            for a in arr:
                if a > pivot:
                    ll.append(a)
                else:
                    rr.append(a)
            return ll, rr

        left, right = divide(array[:pivot_index] + array[pivot_index + 1 :])
        *_, left = quick_sort(left)
        *_, right = quick_sort(right)
        yield left + [pivot] + right


