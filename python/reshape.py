# Flatten the 3D matrix into 1D array
def reshape_to_1d(matrix):
    flattened = []
    depth = len(matrix)
    height = len(matrix[0])
    width = len(matrix[0][0])
    
    for d in range(depth):
        for h in range(height):
            for w in range(width):
                flattened.append(matrix[d][h][w])
    
    return flattened
