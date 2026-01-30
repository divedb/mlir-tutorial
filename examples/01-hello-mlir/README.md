# Hello MLIR Example

This is a basic introduction to MLIR syntax and structure.

## Example 1: Simple Function

File: `simple_function.mlir`

```mlir
// A simple function that returns a constant
func.func @return_constant() -> i32 {
  %c42 = arith.constant 42 : i32
  return %c42 : i32
}
```

### Explanation

- `func.func @return_constant()`: Defines a function named `return_constant`
- `-> i32`: The function returns a 32-bit integer
- `%c42 = arith.constant 42 : i32`: Creates a constant value 42 of type i32
- `return %c42 : i32`: Returns the constant value

## Example 2: Function with Arguments

File: `add_function.mlir`

```mlir
// Function that adds two integers
func.func @add(%arg0: i32, %arg1: i32) -> i32 {
  %result = arith.addi %arg0, %arg1 : i32
  return %result : i32
}
```

### Explanation

- `@add(%arg0: i32, %arg1: i32)`: Function takes two i32 arguments
- `arith.addi`: Addition operation from the arithmetic dialect
- Result type matches return type

## Example 3: Multiple Operations

File: `compute.mlir`

```mlir
// More complex computation
func.func @compute(%a: i32, %b: i32, %c: i32) -> i32 {
  // (a + b) * c
  %sum = arith.addi %a, %b : i32
  %result = arith.muli %sum, %c : i32
  return %result : i32
}
```

## Example 4: Floating Point

File: `float_ops.mlir`

```mlir
// Floating point operations
func.func @distance_squared(%x1: f32, %y1: f32, %x2: f32, %y2: f32) -> f32 {
  // (x2 - x1)^2 + (y2 - y1)^2
  %dx = arith.subf %x2, %x1 : f32
  %dy = arith.subf %y2, %y1 : f32
  %dx2 = arith.mulf %dx, %dx : f32
  %dy2 = arith.mulf %dy, %dy : f32
  %dist2 = arith.addf %dx2, %dy2 : f32
  return %dist2 : f32
}
```

## Running the Examples

### Verify and Print

```bash
mlir-opt simple_function.mlir
```

### Apply Canonicalization

```bash
mlir-opt compute.mlir --canonicalize
```

### Lower to LLVM Dialect

```bash
mlir-opt simple_function.mlir --convert-func-to-llvm --reconcile-unrealized-casts
```

### Translate to LLVM IR

```bash
mlir-translate --mlir-to-llvmir simple_function.mlir
```

## Expected Output

For `simple_function.mlir`, you should see:
```mlir
module {
  func.func @return_constant() -> i32 {
    %c42_i32 = arith.constant 42 : i32
    return %c42_i32 : i32
  }
}
```

## Exercises

1. Modify `add_function.mlir` to compute subtraction instead
2. Create a function that computes the average of three numbers (use f32)
3. Write a function that takes four i32 values and returns their sum
4. Create a function that computes `(a * b) + (c * d)`

## Next Steps

- Learn about [control flow](../02-custom-dialect/)
- Explore [transformations](../03-transformations/)
- Build your own [custom dialect](../../docs/custom-dialects.md)
