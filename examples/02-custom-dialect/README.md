# Custom Dialect Example

This example demonstrates control flow operations in MLIR.

## Example 1: Conditional (If-Else)

File: `conditional.mlir`

```mlir
func.func @abs(%x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %is_negative = arith.cmpi "slt", %x, %c0 : i32
  %result = scf.if %is_negative -> (i32) {
    %neg = arith.subi %c0, %x : i32
    scf.yield %neg : i32
  } else {
    scf.yield %x : i32
  }
  return %result : i32
}
```

### Explanation

- `arith.cmpi "slt"`: Signed less than comparison
- `scf.if`: Structured control flow if statement
- `scf.yield`: Returns a value from the if region

## Example 2: For Loop

File: `loops.mlir`

```mlir
func.func @sum_range(%n: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  %sum = scf.for %i = %c0 to %n step %c1 
    iter_args(%acc = %c0) -> (index) {
    %next = arith.addi %acc, %i : index
    scf.yield %next : index
  }
  
  return %sum : index
}
```

### Explanation

- `scf.for`: Structured for loop
- `iter_args`: Loop-carried values (accumulator)
- Loop iterates from 0 to n (exclusive)

## Example 3: While Loop

File: `while_loop.mlir`

```mlir
func.func @factorial(%n: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  
  %result:2 = scf.while (%i = %n, %acc = %c1) : (i32, i32) -> (i32, i32) {
    %condition = arith.cmpi "sgt", %i, %c0 : i32
    scf.condition(%condition) %i, %acc : i32, i32
  } do {
  ^bb0(%i: i32, %acc: i32):
    %new_acc = arith.muli %acc, %i : i32
    %c1_block = arith.constant 1 : i32
    %new_i = arith.subi %i, %c1_block : i32
    scf.yield %new_i, %new_acc : i32, i32
  }
  
  return %result#1 : i32
}
```

### Explanation

- `scf.while`: While loop with condition checking
- Loop continues while condition is true
- Returns multiple values (iterator and accumulator)

## Example 4: Nested Loops

File: `nested_loops.mlir`

```mlir
func.func @matrix_sum(%rows: index, %cols: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  %sum = scf.for %i = %c0 to %rows step %c1
    iter_args(%outer_sum = %c0) -> (index) {
    %inner_sum = scf.for %j = %c0 to %cols step %c1
      iter_args(%acc = %outer_sum) -> (index) {
      // In a real scenario, this would load from a matrix
      // Here we just add i + j as a placeholder
      %i_plus_j = arith.addi %i, %j : index
      %next_acc = arith.addi %acc, %i_plus_j : index
      scf.yield %next_acc : index
    }
    scf.yield %inner_sum : index
  }
  
  return %sum : index
}
```

## Example 5: Switch/Select

File: `select.mlir`

```mlir
func.func @max(%a: i32, %b: i32) -> i32 {
  %cmp = arith.cmpi "sgt", %a, %b : i32
  %max = arith.select %cmp, %a, %b : i32
  return %max : i32
}

func.func @clamp(%x: i32, %min: i32, %max: i32) -> i32 {
  %too_small = arith.cmpi "slt", %x, %min : i32
  %clamped_low = arith.select %too_small, %min, %x : i32
  %too_large = arith.cmpi "sgt", %clamped_low, %max : i32
  %clamped = arith.select %too_large, %max, %clamped_low : i32
  return %clamped : i32
}
```

### Explanation

- `arith.select`: Ternary operator (condition ? true_val : false_val)
- Useful for simple conditional value selection

## Running the Examples

### Parse and Verify

```bash
mlir-opt conditional.mlir
mlir-opt loops.mlir
```

### Apply Optimizations

```bash
# Canonicalize patterns
mlir-opt loops.mlir --canonicalize

# Lower SCF to CF (control flow)
mlir-opt loops.mlir --convert-scf-to-cf

# Full lowering pipeline
mlir-opt loops.mlir \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts
```

## Exercises

1. Write a function that finds the maximum of three numbers
2. Implement a function that computes the sum of even numbers up to n
3. Create a function that implements the Fibonacci sequence iteratively
4. Write a function that checks if a number is prime (using loops)

## Advanced Topics

- Loop unrolling and vectorization
- Affine loops for better optimization
- Memory operations with loops
- Parallel execution hints

## Next Steps

- Learn about [transformations](../03-transformations/)
- Explore [affine dialect](../../docs/affine-dialect.md)
- Understand [memory operations](../../docs/memory-ops.md)
