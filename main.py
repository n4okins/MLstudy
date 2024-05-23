import random
import time

from algorithm.sorting import (
    bubble_sort,
    comb_sort,
    gnome_sort,
    insertion_sort,
    merge_sort,
    select_sort,
    shaker_sort,
    quick_sort
)

arr = [random.randint(-10, 10) for i in range(20)]
for sorting in [
    bubble_sort,
    shaker_sort,
    comb_sort,
    gnome_sort,
    merge_sort,
    select_sort,
    insertion_sort,
    quick_sort
]:
    s = time.perf_counter()
    *sorted_step, sorted_arr = sorting(arr.copy(), reverse=False)
    t = time.perf_counter() - s
    print(
        f"{sorting.__name__:16s} {t:.8f}[s]", f"Step:{len(sorted_step):4d}", sorted_arr
    )
print(arr)
