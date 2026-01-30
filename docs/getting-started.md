# Getting Started with MLIR

This guide will help you get started with MLIR and understand its core concepts.

## Understanding MLIR Architecture

MLIR is designed around the following key concepts:

### 1. Operations (Ops)

Operations are the fundamental units of computation in MLIR. Every operation has:
- A unique name (e.g., `arith.addi`, `func.func`)
- Zero or more operands (input values)
- Zero or more results (output values)
- Zero or more attributes (compile-time constants)
- Zero or more regions (nested IR)

Example:
```mlir
%result = arith.addi %a, %b : i32
```

### 2. Types

MLIR has a rich type system that includes:
- Built-in types (integers, floats, tensors, vectors)
- Dialect-specific types
- Function types
- Composite types

Example:
```mlir
i32           // 32-bit integer
f64           // 64-bit float
tensor<10xf32> // Tensor of 10 float32 values
memref<10x20xf32> // Memory reference
```

### 3. Attributes

Attributes are compile-time values that provide additional information:
- Integer attributes
- String attributes
- Type attributes
- Array attributes

Example:
```mlir
#myattr = dense<[1, 2, 3, 4]> : tensor<4xi32>
```

### 4. Dialects

Dialects are namespaces for operations, types, and attributes. They enable:
- Domain-specific abstractions
- Progressive lowering
- Modular compiler design

Common dialects:
- `arith`: Arithmetic operations
- `func`: Function definitions
- `scf`: Structured control flow
- `affine`: Affine transformations
- `llvm`: LLVM IR operations

### 5. Regions and Blocks

Regions contain blocks, which contain operations:
- Regions provide hierarchical structure
- Blocks are basic blocks of operations
- Control flow between blocks

Example:
```mlir
func.func @example(%arg0: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %result = arith.addi %arg0, %c1 : i32
  return %result : i32
}
```

## Development Environment Setup

### Required Tools

1. **C++ Compiler**: GCC 7+ or Clang 5+
2. **CMake**: Version 3.13.4 or higher
3. **Ninja**: Fast build system (optional but recommended)
4. **Python**: Version 3.6+ (for build scripts)

### Building LLVM/MLIR

#### Option 1: Full Build

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++

ninja check-mlir
```

#### Option 2: Minimal Build (Faster)

```bash
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON

ninja mlir-opt mlir-translate
```

## Your First MLIR Program

Create a simple MLIR file `hello.mlir`:

```mlir
func.func @main() -> i32 {
  %c42 = arith.constant 42 : i32
  return %c42 : i32
}
```

Run it through `mlir-opt` to verify:

```bash
mlir-opt hello.mlir
```

## Common Tools

### mlir-opt
Optimization and transformation tool
```bash
mlir-opt input.mlir -pass-pipeline='builtin.module(func.func(cse))' -o output.mlir
```

### mlir-translate
Translate between MLIR and other formats
```bash
mlir-translate --mlir-to-llvmir input.mlir -o output.ll
```

### mlir-tblgen
Generate C++ code from TableGen definitions
```bash
mlir-tblgen input.td --gen-op-decls -o ops.h.inc
```

## Next Steps

1. Explore the [Basic Concepts](basic-concepts.md) guide
2. Try the [Hello MLIR](../examples/01-hello-mlir) example
3. Learn about [Custom Dialects](../examples/02-custom-dialect)
4. Understand [Transformations and Passes](../examples/03-transformations)

## Troubleshooting

### Build Issues

**Problem**: CMake can't find LLVM
```bash
# Solution: Set LLVM_DIR
cmake ... -DLLVM_DIR=/path/to/llvm/build/lib/cmake/llvm
```

**Problem**: Out of memory during build
```bash
# Solution: Limit parallel jobs
ninja -j4
```

### Runtime Issues

**Problem**: Dialect not loaded
```mlir
// Solution: Register dialect in your tool
registerDialect<MyDialect>();
```

## Resources

- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [MLIR Tutorials](https://mlir.llvm.org/docs/Tutorials/)
- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
