# MLIR Basic Concepts

This document covers the fundamental concepts you need to understand MLIR.

## The MLIR Hierarchy

```
Module
  └─ Operation (func.func, etc.)
      └─ Region
          └─ Block
              └─ Operation
                  ├─ Operands (values)
                  ├─ Results (values)
                  ├─ Attributes (metadata)
                  └─ Regions (nested)
```

## 1. Values and SSA Form

MLIR uses Static Single Assignment (SSA) form:
- Each value is defined exactly once
- Values are immutable
- Use dominance: a value must be defined before use

Example:
```mlir
func.func @ssa_example(%arg0: i32, %arg1: i32) -> i32 {
  // %arg0 and %arg1 are block arguments (SSA values)
  %0 = arith.addi %arg0, %arg1 : i32  // %0 is defined once
  %1 = arith.muli %0, %arg0 : i32     // %1 uses %0
  return %1 : i32
}
```

## 2. Operations in Detail

### Operation Structure

Every operation has:

```mlir
%results = "dialect.operation"(%operands) {
  attribute_name = attribute_value
} : (operand_types) -> (result_types)
```

Example:
```mlir
%sum = arith.addi %a, %b : i32
```

This is equivalent to the verbose form:
```mlir
%sum = "arith.addi"(%a, %b) : (i32, i32) -> i32
```

### Built-in Operations

Common operations you'll use:

```mlir
// Function definition
func.func @my_function(%arg0: i32) -> i32 {
  return %arg0 : i32
}

// Function call
%result = func.call @my_function(%value) : (i32) -> i32

// Constants
%c0 = arith.constant 0 : i32
%cf = arith.constant 3.14 : f32

// Arithmetic
%add = arith.addi %a, %b : i32
%mul = arith.mulf %x, %y : f32

// Comparisons
%cmp = arith.cmpi "slt", %a, %b : i32  // signed less than
```

## 3. Types System

### Primitive Types

```mlir
// Integer types
i1, i8, i16, i32, i64, i128

// Floating point types
f16, f32, f64

// Index type (target-dependent integer)
index
```

### Aggregate Types

```mlir
// Tensors (immutable)
tensor<10xf32>           // 1D tensor
tensor<10x20xf32>        // 2D tensor
tensor<?x?xf32>          // Dynamic dimensions
tensor<*xf32>            // Unranked tensor

// Memory References (mutable)
memref<10xf32>           // 1D memref
memref<10x20xf32>        // 2D memref
memref<?x?xf32>          // Dynamic dimensions

// Vectors (hardware-friendly)
vector<4xf32>            // 4-element vector
vector<4x4xf32>          // 4x4 2D vector
```

### Function Types

```mlir
(i32, i32) -> i32        // Takes two i32, returns one i32
(tensor<10xf32>) -> ()   // Takes tensor, returns nothing
```

## 4. Dialects

Dialects organize operations into namespaces:

### Common Dialects

**arith** - Arithmetic operations
```mlir
%sum = arith.addi %a, %b : i32
%product = arith.mulf %x, %y : f32
```

**func** - Functions
```mlir
func.func @example() { }
%result = func.call @example() : () -> ()
```

**scf** - Structured Control Flow
```mlir
scf.if %condition {
  // true branch
} else {
  // false branch
}

scf.for %i = %lb to %ub step %step {
  // loop body
}
```

**affine** - Affine transformations
```mlir
affine.for %i = 0 to 10 {
  affine.for %j = 0 to 20 {
    // nested loop with affine guarantees
  }
}
```

**llvm** - LLVM IR operations
```mlir
%ptr = llvm.getelementptr %base[%index] : (!llvm.ptr, i32) -> !llvm.ptr
%val = llvm.load %ptr : !llvm.ptr -> i32
```

## 5. Regions and Blocks

### Regions

Regions contain blocks and provide isolation:

```mlir
// Function region contains one or more blocks
func.func @example(%arg: i32) -> i32 {
  // This is block ^bb0 (entry block)
  return %arg : i32
}
```

### Multiple Blocks

```mlir
func.func @branches(%cond: i1, %a: i32, %b: i32) -> i32 {
^bb0:
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%a : i32)
^bb2:
  cf.br ^bb3(%b : i32)
^bb3(%result: i32):
  return %result : i32
}
```

## 6. Attributes

Attributes are compile-time values:

```mlir
// Integer attributes
#int_attr = 42 : i32

// String attributes
#str_attr = "hello"

// Array attributes
#array_attr = [1, 2, 3, 4]

// Dense attributes (for tensors)
#dense_attr = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>

// Type attributes
#type_attr = i32
```

Usage in operations:
```mlir
%c = arith.constant 42 : i32
"my.op"() { attribute = 42 : i32 } : () -> ()
```

## 7. Traits and Interfaces

### Traits

Traits describe operation properties:
- `NoSideEffect`: Operation is pure
- `Commutative`: Operands can be swapped
- `Terminator`: Ends a block

### Interfaces

Interfaces provide common APIs:
- `MemoryEffectOpInterface`: Describes memory effects
- `InferTypeOpInterface`: Infers result types
- `LoopLikeOpInterface`: Common loop operations

## 8. Pattern Matching and Rewriting

MLIR provides powerful pattern matching:

```cpp
// Example rewrite pattern in C++
struct SimplifyAddZero : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    // Match: x + 0
    if (isConstantZero(op.getRhs())) {
      // Rewrite: replace with x
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    return failure();
  }
};
```

## 9. Conversion Framework

MLIR supports systematic dialect conversion:

1. **Legality**: Define which operations are legal
2. **Patterns**: Define how to convert operations
3. **Type Conversion**: Convert between type systems

Example conversion flow:
```
High-level Dialect
  ↓ (lowering pass)
Mid-level Dialect
  ↓ (lowering pass)
LLVM Dialect
  ↓ (translation)
LLVM IR
```

## 10. Verification

MLIR enforces correctness:
- Type checking
- SSA form validation
- Trait verification
- Custom verifiers

Example:
```cpp
LogicalResult MyOp::verify() {
  // Custom verification logic
  if (getOperand().getType() != getResult().getType())
    return emitError("operand and result types must match");
  return success();
}
```

## Best Practices

1. **Use SSA form correctly**: Define values before use
2. **Choose appropriate dialects**: Use the right abstraction level
3. **Verify your IR**: Always run verification
4. **Document operations**: Add descriptions to custom ops
5. **Test transformations**: Ensure passes preserve semantics

## Next Steps

- Try the [examples](../examples/)
- Read the [MLIR Tutorials](https://mlir.llvm.org/docs/Tutorials/)
- Learn about [Creating Dialects](https://mlir.llvm.org/docs/DefiningDialects/)

## References

- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [MLIR Dialect Definitions](https://mlir.llvm.org/docs/Dialects/)
- [MLIR Rationale](https://mlir.llvm.org/docs/Rationale/)
