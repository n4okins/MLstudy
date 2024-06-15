count = 0
n = iter(lambda: count, None)
for _ in range(5):
    print(next(n))
    count += 1