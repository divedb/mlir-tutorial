# Contributing to MLIR Tutorial

Thank you for your interest in contributing to this MLIR tutorial! This document provides guidelines for contributing.

## How to Contribute

### Reporting Issues

If you find errors, unclear explanations, or missing content:

1. Check if the issue already exists
2. Create a new issue with:
   - Clear description of the problem
   - Location (file and line number if applicable)
   - Suggested improvement (optional)

### Suggesting Enhancements

For new tutorial content or improvements:

1. Open an issue describing:
   - What you'd like to add/change
   - Why it would be valuable
   - How it fits into the existing structure

### Contributing Code

#### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Test your examples (ensure they parse correctly)
5. Update documentation as needed
6. Commit with clear messages
7. Push to your fork
8. Submit a pull request

#### Example Guidelines

When adding MLIR examples:

1. **Correctness**: Examples must be valid MLIR
   ```bash
   mlir-opt your-example.mlir --verify-diagnostics
   ```

2. **Comments**: Add explanatory comments
   ```mlir
   // This function demonstrates constant folding
   func.func @example() -> i32 {
     %c42 = arith.constant 42 : i32  // Define constant
     return %c42 : i32                // Return it
   }
   ```

3. **Documentation**: Include README explaining:
   - What the example demonstrates
   - How to run it
   - Expected output
   - Key concepts

4. **Naming**: Use descriptive names
   - Files: `simple_function.mlir`, `loop_optimization.mlir`
   - Functions: `@compute_distance`, `@matrix_multiply`

#### Documentation Guidelines

1. **Clear and concise**: Explain concepts simply
2. **Code examples**: Include runnable code
3. **Step-by-step**: Break complex topics into steps
4. **References**: Link to official MLIR docs
5. **Formatting**: Use markdown consistently

#### Directory Structure

```
mlir-tutorial/
├── README.md              # Main overview
├── docs/                  # Detailed documentation
│   ├── getting-started.md
│   └── basic-concepts.md
└── examples/              # Code examples
    ├── 01-hello-mlir/
    │   ├── README.md
    │   └── *.mlir
    └── 02-custom-dialect/
        ├── README.md
        └── *.mlir
```

### Testing Examples

Before submitting, verify your examples:

```bash
# Parse and verify
mlir-opt example.mlir

# Test transformations
mlir-opt example.mlir --canonicalize

# Check LLVM conversion
mlir-opt example.mlir \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts
```

## Style Guide

### MLIR Code Style

1. **Indentation**: 2 spaces
2. **Line length**: Aim for 80 characters
3. **Comments**: Use `//` for single-line comments
4. **Naming**: 
   - Functions: `@snake_case`
   - Values: `%snake_case`
   - Constants: `%c<value>` (e.g., `%c0`, `%c1`)

Example:
```mlir
func.func @well_formatted(%arg0: i32, %arg1: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %sum = arith.addi %arg0, %arg1 : i32
  %result = arith.addi %sum, %c1 : i32
  return %result : i32
}
```

### Markdown Style

1. **Headers**: Use ATX style (`#`, `##`, `###`)
2. **Code blocks**: Specify language
   ````markdown
   ```mlir
   func.func @example() { }
   ```
   ````
3. **Lists**: Use `-` for unordered, `1.` for ordered
4. **Links**: Use reference-style for repeated links

## Community

### Communication

- **Issues**: For bugs and feature requests
- **Pull Requests**: For contributing code
- **Discussions**: For questions and ideas

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn

## Recognition

Contributors will be acknowledged in the project. Significant contributions may be highlighted in release notes.

## Questions?

If you're unsure about anything, feel free to:
- Open an issue asking for clarification
- Start a discussion
- Look at existing examples and documentation

Thank you for helping make MLIR more accessible to everyone!
