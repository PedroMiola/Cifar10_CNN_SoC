def maxpool(matrix, size, stride):
    rows = len(matrix)
    cols = len(matrix[0])
    out = []

    for i in range(0, rows - size + 1, stride):
        row_pool = []
        for j in range(0, cols - size + 1, stride):
            window = [matrix[x][j:j+size] for x in range(i, i+size)]
            max_val = max(map(max, window))
            row_pool.append(max_val)
        out.append(row_pool)

    return out


m = [
    [1, 3, 2, 1],
    [4, 6, 5, 2],
    [7, 8, 9, 4],
    [1, 2, 0, 3]
]

print(maxpool(m, size=2, stride=2))