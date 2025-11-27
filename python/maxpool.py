import math

def _get_shape(nested_list):
    shape = []
    current_level = nested_list
    while isinstance(current_level, list):
        shape.append(len(current_level))
        if len(current_level) > 0:
            current_level = current_level[0]
        else:
            break
    return tuple(shape)


def maxpool(matrix_3d, size, stride):
    """
    Apply 2D max pooling to a 3D input matrix [rows][cols][channels]
    
    Args:
        matrix_3d: 3D list of shape [rows][cols][channels]
        size: pooling window size
        stride: stride for pooling operation
    
    Returns:
        3D list of shape [out_rows][out_cols][channels]
    """
    rows = len(matrix_3d)
    cols = len(matrix_3d[0])
    channels = len(matrix_3d[0][0])
    
    # Calculate output dimensions with padding
    out_rows = math.ceil(rows / stride)
    out_cols = math.ceil(cols / stride)
    
    # Calculate padding needed
    pad_rows = max(0, (out_rows - 1) * stride + size - rows)
    pad_cols = max(0, (out_cols - 1) * stride + size - cols)
    
    # Initialize output tensor
    out = []
    
    # Process each channel separately
    for channel in range(channels):
        # Extract 2D matrix for this channel
        channel_matrix = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(matrix_3d[i][j][channel])
            channel_matrix.append(row)
        
        # Pad the channel matrix
        if pad_rows > 0:
            padding_row = [-float('inf')] * cols
            for _ in range(pad_rows):
                channel_matrix.append(padding_row)
        
        if pad_cols > 0:
            padded_cols = len(channel_matrix[0])
            for i in range(len(channel_matrix)):
                channel_matrix[i].extend([-float('inf')] * pad_cols)
        
        # Apply 2D max pooling to this channel
        padded_rows = len(channel_matrix)
        padded_cols = len(channel_matrix[0])
        
        channel_out = []
        for i in range(0, padded_rows - size + 1, stride):
            row_pool = []
            for j in range(0, padded_cols - size + 1, stride):
                window = [channel_matrix[x][j:j+size] for x in range(i, i+size)]
                max_val = max(map(max, window))
                row_pool.append(max_val)
            channel_out.append(row_pool)
        
        # Store results for this channel
        out.append(channel_out)
    
    # Reorganize output to maintain [rows][cols][channels] order
    result = []
    for i in range(len(out[0])):  # rows
        result_row = []
        for j in range(len(out[0][0])):  # cols
            channel_vals = []
            for channel in range(channels):
                channel_vals.append(out[channel][i][j])
            result_row.append(channel_vals)
        result.append(result_row)
    
    return result