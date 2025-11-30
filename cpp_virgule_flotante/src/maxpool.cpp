#include "../include/cnn.hpp"

matrix3D maxpool(matrix3D& matrix, int size=3, int stride=2) {

    int rows,cols,channels;

    int out_rows = std::ceil(rows/stride);
    int out_cols = std::ceil(cols/stride);

    // Padding needed
    int pad_rows = std::max(0, (out_rows - 1) * stride + size - rows);
    int pad_cols = std::max(0, (out_cols - 1) * stride + size - cols);

    matrix3D<out_rows,out_cols,channels> temp_mtx; // intermediary channel-major representation

    for (int c = 0; c < channels; c++) {
        // extract individual channel matrices
        matrix2D<rows, cols> channel_matrix;

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                channel_matrix[i][j] = temp_mtx[i][j][c];    

        data_t local_max;
        for (int i = 0; i < out_rows; i++) {
            for (int j = 0; j < out_cols; j++) { // 3x3 window, corresponding to a value in the final matrix
                local_max = 0;
                for (int m = i - std::floor(size/2); m < size; m++) { // search window on original matrix and take the maximum
                    for (int n = j - std::floor(size/2); n < size; n++) {
                        if (channel_matrix[m+1][n+1]>local_max) {
                            local_max = channel_matrix[m+1][n+1];
                        }
                        out_mtx[i][j][c] = local_max;
                    }
                }
            }
        }

    }



    return out_mtx;
}


    for (int c = 0; c < channels; c++) {
        // Perform max pooling
        vector<vector<float>> channel_out;

        for (int i = 0; i <= padded_rows - size; i += stride) {
            vector<float> row_pool;
            for (int j = 0; j <= padded_cols - size; j += stride) {
                float max_val = 0;
                for (int x = i; x < i + size; x++) {
                    for (int y = j; y < j + size; y++) {
                        max_val = std::max(max_val, channel_matrix[x][y]);
                    }
                }
                row_pool.push_back(max_val);
            }
            channel_out.push_back(row_pool);
        }

        out.push_back(channel_out);
    }

    // Reorganize to [rows][cols][channels]
    vector<vector<vector<float>>> result(out_rows, vector<vector<float>>(out_cols, vector<float>(channels)));

    for (int i = 0; i < out_rows; i++) {
    for (int j = 0; j < out_cols; j++) {
        for (int c = 0; c < channels; c++) {
            out_mtx[i][j][c] = temp_mtx[c][i][j];
        }
    }


    }



/*

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
*/