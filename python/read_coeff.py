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


def write_coefficients(file_path, coefficients):
    """
    Write the coefficients to a file in the specified format.
    coefficients_dict should be a dictionary where keys are tensor names
    and values are nested lists or numpy arrays.
    """
    with open(file_path, "wb") as f:
        for tensor_name, value in coefficients:
            if tensor_name == "conv1/biases":
                # Write the values in binary format skipping the tensor_name line
                f.write(value.tobytes())
                print(f'Wrote tensor {tensor_name} with values {value}')


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

#read_coefficients_path = '../cifar10_coeffs/CNN_coeff_3x3.txt'
#output_path = '../cifar10_coeffs/CNN_bin_coeff_3x3.bin'

#out = read_coefficients(read_coefficients_path, as_numpy=True)
#write_coefficients(output_path, out)



import re
from pathlib import Path

def _sanitize_name(name: str) -> str:
    """
    Turn something like 'layer1.weight' into a valid C++ identifier: 'layer1_weight'.
    """
    return re.sub(r'[^0-9a-zA-Z_]', '_', name)

def _infer_shape_and_flatten(value):
    """
    Supports:
      - NumPy arrays
      - PyTorch tensors
      - Nested Python lists/tuples
    Returns (shape_tuple, flat_list).
    """
    # Handle NumPy / torch if available
    try:
        import numpy as np
    except ImportError:
        np = None

    # PyTorch tensor -> NumPy
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()

    # Generic NumPy array
    if np is not None and isinstance(value, np.ndarray):
        return value.shape, value.flatten().tolist()

    # Fallback: nested Python lists/tuples
    def infer_shape(x):
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return (0,)
            return (len(x),) + infer_shape(x[0])
        else:
            return ()

    def flatten(x):
        if isinstance(x, (list, tuple)):
            for item in x:
                yield from flatten(item)
        else:
            yield x

    shape = infer_shape(value)
    flat = list(flatten(value))
    return shape, flat

def _flat_to_cpp_initializer(flat_values, dtype: str) -> str:
    """
    Convert a flat list of numbers into a C++ initializer list string.
    """
    elems = []
    float_like = dtype in ("float", "double")
    for v in flat_values:
        if float_like:
            elems.append(f"{float(v):.8f}{'f' if dtype=='float' else ''}")
        else:
            elems.append(str(int(v)))
    return "{ " + ", ".join(elems) + " }"

def export_tensors_to_header(tensor_list, header_path: str,
                             namespace: str = "tensors",
                             dtype: str = "float") -> None:
    """
    tensor_list: list of (name, values)
      - name: string (e.g. 'fc1.weight')
      - values: NumPy array, torch.Tensor, or nested lists/tuples

    header_path: output .hpp path
    namespace: C++ namespace to wrap everything in
    dtype: 'float', 'double', or an integer type
    """
    header_path = Path(header_path)
    guard = re.sub(r'[^0-9A-Z_]', '_', header_path.name.upper())

    lines = []
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append("#include <cstddef>")
    lines.append("")

    if namespace:
        lines.append(f"namespace {namespace} {{")
        lines.append("")

    for name, values in tensor_list:
        cpp_name = _sanitize_name(name)
        shape, flat = _infer_shape_and_flatten(values)
        initializer = _flat_to_cpp_initializer(flat, dtype=dtype)

        # Emit dimension constants
        for i, dim in enumerate(shape):
            lines.append(f"static const std::size_t {cpp_name}_dim{i} = {dim};")

        if not shape:  # scalar
            scalar_value = initializer.strip("{ }")
            lines.append(f"static const {dtype} {cpp_name} = {scalar_value};")
        else:
            # e.g. float W[3][4][5] = { ... };
            dims_str = "".join(f"[{d}]" for d in shape)
            lines.append(f"static const {dtype} {cpp_name}{dims_str} = {initializer};")

        lines.append("")

    if namespace:
        lines.append(f"}} // namespace {namespace}")
        lines.append("")

    lines.append(f"#endif // {guard}")
    lines.append("")

    header_path.write_text("\n".join(lines), encoding="utf-8")


out = read_coefficients('../cifar10_coeffs/CNN_coeff_3x3.txt', as_numpy=True)
export_tensors_to_header(
    out,
    header_path='../cpp_virgule_flotante/include/cnn_coefficients.hpp',
    namespace='cnn_coefficients',
    dtype='float'
)