# Transformations and Passes

This example demonstrates MLIR transformations and optimization passes.

## Overview

MLIR provides a powerful framework for transforming IR through:
- Pattern rewriting
- Canonicalization
- Dialect conversion
- Custom passes

## Example 1: Canonicalization

Canonicalization simplifies operations to their canonical form.

### Input: `before_canon.mlir`

```mlir
func.func @redundant_ops(%x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  
  // x + 0 = x (identity)
  %add_zero = arith.addi %x, %c0 : i32
  
  // x * 1 = x (identity)
  %mul_one = arith.muli %add_zero, %c1 : i32
  
  return %mul_one : i32
}
```

### Run Canonicalization

```bash
mlir-opt before_canon.mlir --canonicalize -o after_canon.mlir
```

### Expected Output

```mlir
func.func @redundant_ops(%x: i32) -> i32 {
  return %x : i32
}
```

The canonicalization pass recognizes:
- `x + 0 = x`
- `x * 1 = x`

## Example 2: Common Subexpression Elimination (CSE)

CSE removes redundant computations.

### Input: `before_cse.mlir`

```mlir
func.func @duplicate_computation(%a: i32, %b: i32) -> i32 {
  %sum1 = arith.addi %a, %b : i32
  %prod1 = arith.muli %sum1, %sum1 : i32
  
  // Same computation again
  %sum2 = arith.addi %a, %b : i32
  %prod2 = arith.muli %sum2, %sum2 : i32
  
  %result = arith.addi %prod1, %prod2 : i32
  return %result : i32
}
```

### Run CSE

```bash
mlir-opt before_cse.mlir --cse -o after_cse.mlir
```

### Expected Output

```mlir
func.func @duplicate_computation(%a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  %prod = arith.muli %sum, %sum : i32
  %result = arith.addi %prod, %prod : i32
  return %result : i32
}
```

## Example 3: Constant Folding

Evaluates constant expressions at compile time.

### Input: `constant_expr.mlir`

```mlir
func.func @constant_computations() -> i32 {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  
  // (2 + 3) * 4 = 5 * 4 = 20
  %sum = arith.addi %c2, %c3 : i32
  %result = arith.muli %sum, %c4 : i32
  
  return %result : i32
}
```

### Run Canonicalization (includes constant folding)

```bash
mlir-opt constant_expr.mlir --canonicalize
```

### Expected Output

```mlir
func.func @constant_computations() -> i32 {
  %c20_i32 = arith.constant 20 : i32
  return %c20_i32 : i32
}
```

## Example 4: Loop Optimizations

### Input: `loop_optimize.mlir`

```mlir
func.func @simple_loop(%n: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  %result = scf.for %i = %c0 to %n step %c1
    iter_args(%acc = %c0) -> (index) {
    // acc = acc + 2
    %next = arith.addi %acc, %c2 : index
    scf.yield %next : index
  }
  
  return %result : index
}
```

### Apply Loop-Invariant Code Motion

```bash
mlir-opt loop_optimize.mlir --loop-invariant-code-motion
```

## Example 5: Dialect Conversion

Converting from high-level to low-level dialects.

### Input: `high_level.mlir`

```mlir
func.func @convert_example(%a: i32, %b: i32) -> i32 {
  %result = arith.addi %a, %b : i32
  return %result : i32
}
```

### Convert SCF to Control Flow

```bash
mlir-opt high_level.mlir --convert-scf-to-cf
```

### Convert to LLVM Dialect

```bash
mlir-opt high_level.mlir \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts
```

### Translate to LLVM IR

```bash
mlir-translate --mlir-to-llvmir high_level.mlir -o output.ll
```

## Common Passes

### Optimization Passes

```bash
# Canonicalization
mlir-opt input.mlir --canonicalize

# Common Subexpression Elimination
mlir-opt input.mlir --cse

# Dead Code Elimination
mlir-opt input.mlir --symbol-dce

# Inlining
mlir-opt input.mlir --inline

# Loop optimizations
mlir-opt input.mlir --loop-invariant-code-motion
```

### Lowering Passes

```bash
# Convert SCF to Control Flow
mlir-opt input.mlir --convert-scf-to-cf

# Convert to LLVM Dialect
mlir-opt input.mlir \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts

# Full lowering pipeline
mlir-opt input.mlir \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts
```

### Analysis Passes

```bash
# Print IR with dominance information
mlir-opt input.mlir --view-op-graph

# Print statistics
mlir-opt input.mlir --mlir-print-op-stats
```

## Pass Pipelines

Combine multiple passes:

```bash
mlir-opt input.mlir \
  --pass-pipeline='builtin.module(func.func(cse,canonicalize))'
```

## Creating Custom Passes

Example of a simple custom pass (C++):

```cpp
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"

namespace {
struct MyOptimizationPass : 
    public PassWrapper<MyOptimizationPass, OperationPass<func::FuncOp>> {
  
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    
    // Apply patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<MyPattern>(&getContext());
    
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};
}

std::unique_ptr<Pass> createMyOptimizationPass() {
  return std::make_unique<MyOptimizationPass>();
}
```

## Exercises

1. Create an MLIR file with redundant operations and apply canonicalization
2. Write a function with duplicate subexpressions and run CSE
3. Create a pipeline that applies multiple optimization passes
4. Lower a simple function all the way to LLVM IR

## Best Practices

1. **Order matters**: Some passes enable others
2. **Iterate**: Run passes multiple times for maximum effect
3. **Verify**: Always verify IR after transformations
4. **Test**: Write tests for your passes
5. **Document**: Explain what your passes do

## Next Steps

- Learn about [pattern rewriting](../../docs/pattern-rewriting.md)
- Explore [custom passes](../../docs/custom-passes.md)
- Study [dialect conversion](../../docs/dialect-conversion.md)

## References

- [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
- [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
- [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
