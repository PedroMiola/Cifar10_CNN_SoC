import numpy as np

def _parse_nested_array(lines):
    """
    Parse a list of lines containing something like:
    [[[[ 1.0 2.0 ]
       [ 3.0 4.0 ]]]]
    into a numpy array with the correct shape.
    """
    text = " ".join(lines)

    stack = []
    current_list = None
    root = None
    num_buf = ""
    valid_num_chars = set("0123456789+-.eE")

    for ch in text:
        if ch in "[]":
            # finish any current number before processing bracket
            if num_buf:
                current_list.append(float(num_buf))
                num_buf = ""
            if ch == "[":
                # start a new list
                if current_list is not None:
                    stack.append(current_list)
                current_list = []
                if root is None:
                    root = current_list
            else:  # ch == ']'
                # close current list and attach to parent if any
                if stack:
                    parent = stack.pop()
                    parent.append(current_list)
                    current_list = parent
        elif ch in valid_num_chars:
            num_buf += ch
        else:
            # whitespace / other separators
            if num_buf:
                current_list.append(float(num_buf))
                num_buf = ""

    # in case the last char was part of a number
    if num_buf:
        current_list.append(float(num_buf))

    return np.array(root, dtype=float)


def read_coefficients(file_path, as_numpy: bool = False):
    """
    Generator that yields (tensor_name, tensor_value).

    - If as_numpy=True  → tensor_value is a np.ndarray
    - If as_numpy=False → tensor_value is a nested Python list (list of lists)
                           that you can index like t[i][j][k][l]
    """
    with open(file_path, "r") as f:
        tensor_name = None
        collecting = False
        buffer_lines = []
        bracket_level = 0

        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue

            # New tensor header
            if stripped.startswith("tensor_name:"):
                tensor_name = stripped.split(":", 1)[1].strip()
                collecting = False
                buffer_lines = []
                bracket_level = 0
                continue

            # After we see a tensor_name, collect its bracket block
            if tensor_name is not None:
                if "[" in stripped or collecting:
                    collecting = True
                    buffer_lines.append(stripped)

                    # Track how many [ and ] we've seen to know when the block ends
                    bracket_level += stripped.count("[") - stripped.count("]")

                    if bracket_level == 0 and collecting:
                        # Finished this tensor
                        arr = _parse_nested_array(buffer_lines)
                        value = arr if as_numpy else arr.tolist()
                        yield tensor_name, value

                        # Reset for next tensor
                        tensor_name = None
                        collecting = False
                        buffer_lines = []
                        bracket_level = 0
    return

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


def write_coefficients(file_path, coefficients_dict):
    """
    Write the coefficients to a file in the specified format.
    coefficients_dict should be a dictionary where keys are tensor names
    and values are nested lists or numpy arrays.
    """
    with open(file_path, "w") as f:
        for tensor_name, value in coefficients_dict.items():
            f.write(f"tensor_name: {tensor_name}\n")
            f.write("[\n")
            _write_nested_list(f, value, indent_level=1)
            f.write("]\n\n")

def _write_nested_list(f, nested_list, indent_level):
    indent = "  " * indent_level
    if isinstance(nested_list, list):
        f.write(f"{indent}[\n")
        for item in nested_list:
            _write_nested_list(f, item, indent_level + 1)
        f.write(f"{indent}]\n")
    else:
        f.write(f"{indent}{nested_list}\n")
        
# Example usage:
# path = '../cifar10_coeffs/CNN_coeff_3x3.txt'
# results = read_coefficients(path, as_numpy=False)
# for tensor_name, coefficients in results:
#     print(f'Tensor Name: {tensor_name}')
#     print(f'Shape: {_get_shape(coefficients)}')
#     #print(f'Coefficients: {coefficients[:10]}...')  # Print first 10 coefficients for brevity

#write_coefficients_path = 'output_coefficients.txt'
#coefficients_dict = {
#    tensor_name: coefficients
#    for tensor_name, coefficients in read_coefficients(path, as_numpy=False)
#}

#write_coefficients(write_coefficients_path, coefficients_dict)