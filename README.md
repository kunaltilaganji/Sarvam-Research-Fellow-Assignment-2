
### Sarvam Research Fellow Assignment-2

## Einops `rearrange` Implementation from Scratch

The following Python code provides a standalone implementation of the core functionality found in the `rearrange` operation of the popular `einops` library. It uses only NumPy and standard Python libraries.

## Features:

* **Reshaping:** Implicitly handled through pattern definition.
* **Transposition:** Changing the order of axes names (e.g., `h w -> w h`).
* **Splitting Axes:** Decomposing a dimension into multiple (e.g., `(h w) -> h w` with `h` provided).
* **Merging Axes:** Combining multiple dimensions into one (e.g., `a b -> (a b)`).
* **Repeating Axes:** Creating a new dimension by repeating an existing dimension of size 1 (e.g., `a 1 c -> a b c` with `b` provided).
* **Ellipsis (`...`):** Handling arbitrary leading, middle, or trailing dimensions.
* **Error Handling:** Provides informative error messages for invalid patterns, shape mismatches, and incorrect dimension specifications.

## How it Works:

1.  **Parsing:** The `pattern` string is split into left (input) and right (output) sides. Each side is parsed to identify individual axes, composite axes (in parentheses), and the ellipsis (`...`). Regular expressions help identify these components. Anonymous axes of size 1 (`1`) are tracked internally with unique names.
2.  **Input Analysis:** The parsed left side is matched against the input `tensor.shape`.
    * Ellipsis (`...`) captures the dimensions not explicitly named.
    * Composite axes like `(h w)` are handled. If one sub-axis length (e.g., `h`) is provided in `axes_lengths`, the other (`w`) is inferred by division. If neither is provided but the total dimension matches the product of known lengths from `axes_lengths`, it's validated. If multiple are unknown, an error is raised.
    * All axis lengths (explicit, inferred, ellipsis, internal '1's) are stored. Consistency checks ensure the pattern matches the tensor's rank and dimensions.
3.  **Intermediate Representation:** The core idea is to transform the input tensor into a state where every *elementary* axis (including those created by splitting and internal names for '1') corresponds to a distinct dimension, ordered as they appear in the flattened left pattern. This involves:
    * An initial `reshape` to handle any splitting (e.g., shape `(12, 5)` with pattern `(h w) c` and `h=3` becomes `(3, 4, 5)`).
    * A `transpose` operation to reorder these elementary dimensions according to their sequence in the input pattern string (e.g., if the pattern was `c (h w)`, the intermediate tensor dimensions would correspond to `c, h, w`).
4.  **Output Analysis:** The parsed right side is analyzed.
    * Axes appearing only on the right side *must* be defined in `axes_lengths` and signal a *repetition*.
    * Repetition requires finding a corresponding axis of size 1 on the *input* side (represented by internal names like `_literal_1_...` or named axes confirmed to have size 1). Available size-1 axes are consumed as needed.
    * Checks ensure all necessary input axes are used in the output pattern (either directly or as sources for repetition).
5.  **Final Operations:**
    * **Repetition:** If repetitions are needed, `np.expand_dims` is used to create a new dimension, `np.repeat` duplicates it along that new dimension, and the original source dimension is `np.squeeze`d out. The internal tracking list of axes (`final_target_axes_order`) is updated by inserting the new axis name and removing the source axis name.
    * **Permutation:** A final `transpose` is calculated to reorder the (potentially repeated and squeezed) dimensions according to the elementary axes order derived from the right pattern.
    * **Reshape:** A final `reshape` handles any merging operations specified by parentheses on the right side (e.g., `a b -> (a b)`).

## How to Run:

1.  **Code:** Place the Python code containing the `rearrange` function and the `TestRearrangeScratch` class in a Python file (e.g., `my_einops.py`) or a notebook cell.
2.  **Tests:**
    * In a separate cell or script section, import `unittest` and the test class.
    * Use `unittest.TestLoader().loadTestsFromTestCase(TestRearrangeScratch)` to create a test suite.
    * Use a `TextTestRunner` to execute them:
        ```python
        import unittest
        # from my_einops import TestRearrangeScratch # If using a separate file

        suite = unittest.TestSuite()
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestRearrangeScratch))
        runner = unittest.TextTestRunner(verbosity=2) # More output
        runner.run(suite)
        ```
3.  **Usage:** Import the `rearrange` function and use it as shown in the examples:
    ```python
    import numpy as np
    # from my_einops import rearrange # If using a file

    x = np.random.rand(12, 10)
    result = rearrange(x, '(h w) c -> h w c', h=3)
    print(result.shape)
    ```


## Results:
<p align="center">
  <img src=/workspaces/Sarvam-Research-Fellow-Assignment-2/results.png>
</p>


## Design Decisions:

* **Internal Names for '1':** Using internal names (`_literal_1_...`) helps track anonymous dimensions of size 1, especially when multiple exist or when they are used as sources for repetition.
* **Intermediate State:** Using an intermediate transposed state where dimensions match the elementary input axes (including internal names) simplifies mapping to the output permutation.
* **Repetition Logic:** The updated repetition logic explicitly expands, repeats, and then squeezes the source dimension, ensuring the tensor shape and axis tracking list remain consistent for the final permutation.
* **Error Messages:** Custom `EinopsError` and detailed messages aim to make debugging easier.
* **Parsing:** Regex is used for initial component identification, followed by checks for validity. Improved checks catch more syntax errors.
* **Optimization:** While not heavily micro-optimized, the approach avoids unnecessary intermediate array copies where possible by chaining `reshape` and `transpose`. The main cost comes from these NumPy operations themselves. Parsing is done once.
