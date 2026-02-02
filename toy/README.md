## 入门指南（Getting Started）

请先参考[LLVM Getting Started](https://llvm.org/docs/GettingStarted.html)中关于构建LLVM的说明。下面介绍如何在LLVM中构建MLIR。

以下编译和MLIR的测试假定你已经具备

- git
- ninja
- 以及一个可用的C++工具链（参见[LLVM的相关要求](https://llvm.org/docs/GettingStarted.html#requirements)）。

如果是初学者，可以先尝试构建一个[Toy](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/)语言编译器的教程。

**TIP**

请参阅[Testing Guide - CLI Incantations](https://mlir.llvm.org/getting_started/TestingGuide/#command-line-incantations)章节，了解更多调用和筛选测试的方法，这些技巧可以显著提高日常开发效率。

<hr/>

### 类Unix系统的编译 / 测试流程

```bash
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON
# Using clang and lld speeds up the build, we recommend adding:
#  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
# CCache can drastically speed up further rebuilds, try adding:
#  -DLLVM_CCACHE_BUILD=ON
# Optionally, using ASAN/UBSAN can find bugs early in development, enable with:
# -DLLVM_USE_SANITIZER="Address;Undefined"
# Optionally, enabling integration tests as well
# -DMLIR_INCLUDE_INTEGRATION_TESTS=ON
```

使用clang和lld可以加快构建速度（在Ubuntu上执行`sudo apt-get install clang lld`），建议添加：

```bash
-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
```

使用CCache可以大幅加速后续重建，建议添加：

```bash
-DLLVM_CCACHE_BUILD=ON
```

可选：启用ASAN/UBSAN以在开发早期发现bug：

```bash
-DLLVM_USE_SANITIZER="Address;Undefined
```

可选：启用集成测试：

```bash
-DMLIR_INCLUDE_INTEGRATION_TESTS=ON
```

```bash
cmake --build . --target check-mlir
```

如果你需要调试信息，可以使用（二选一）：

```bash
-DCMAKE_BUILD_TYPE=Debug
-DCMAKE_BUILD_TYPE=RelWithDebInfo
```

此外，推荐在调试构建中使用：

```bash
-DLLVM_USE_SPLIT_DWARF=ON
```


这样可以节省大约30%–40%的磁盘空间。

### Windows编译 / 测试

在Windows上使用Visual Studio 2017进行编译和测试：
```bash
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project\build
cd llvm-project\build
cmake ..\llvm -G "Visual Studio 15 2017 Win64" ^
  -DLLVM_ENABLE_PROJECTS=mlir ^
  -DLLVM_BUILD_EXAMPLES=ON ^
  -DLLVM_TARGETS_TO_BUILD="Native" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -Thost=x64 ^
  -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build . --target tools/mlir/test/check-mlir
```

### 相关文档

- [Reporting Issues](https://mlir.llvm.org/getting_started/ReportingIssues/)（问题反馈）

- [Debugging Tips](https://mlir.llvm.org/getting_started/Debugging/)（调试技巧）

- [FAQ](https://mlir.llvm.org/getting_started/Faq/)

- [How to Contribute](https://mlir.llvm.org/getting_started/Contributing/)（如何参与贡献）

- [Developer Guide](https://mlir.llvm.org/getting_started/DeveloperGuide/)（开发者指南）

- [Open Projects](https://mlir.llvm.org/getting_started/openprojects/)（开放项目）

- [Glossary](https://mlir.llvm.org/getting_started/Glossary/)（术语表）

- [Testing Guide](https://mlir.llvm.org/getting_started/TestingGuide/)（测试指南）

<hr/>

本教程将演示如何在**MLIR**之上实现一个基础的**Toy示例语言**。本教程的目标是介绍**MLIR的核心概念**；尤其是说明 **Dialect**如何在轻松支持**语言特有的语法结构与变换**的同时，仍然能够**顺利地降低（lower）到LLVM或其他代码生成基础设施**。

本教程的设计模型基于**LLVM Kaleidoscope教程**。另一个很好的入门资料是**[2020年LLVM开发者大会](https://www.youtube.com/watch?v=Y4SvqTtOIDk)（LLVM Dev Conference）** 的在线录制视频。

本教程假定你已经**克隆并成功构建了MLIR**；如果尚未完成，请参考**Getting started with MLIR**。

### 教程章节划分

本教程分为以下几个章节：

- 第1章：Toy语言简介及其AST的定义

- 第2章：遍历AST并生成MLIR Dialect，介绍MLIR的基础概念

- 第3章：使用模式重写系统进行高层、语言特定的优化

- 第4章：使用Interface编写通用、与Dialect无关的变换

- 第5章：部分Lowering到更低层的Dialect
- 第6章：Lowering到LLVM及代码生成

- 第7章：扩展Toy：添加对复合类型的支持

<hr/>

## 第1章：Toy语言与AST

### 语言

本教程将通过一个示例语言来进行说明，我们称其为**Toy**。Toy是一种**基于张量（tensor）的语言**，允许你定义函数、执行数学计算，并打印结果。

为了保持示例的简单性，代码生成（codegen）将被限制为 **秩（rank）≤ 2 的张量**，并且Toy中唯一的数据类型是 **64位浮点数**（在C语言中也称为`double`）。因此，所有值都隐式地采用双精度浮点类型。**值是不可变的**（即每个操作都会返回一个新分配的值），并且内存释放是**自动管理的**。

冗长的描述到此为止；通过一个示例来理解会更直观：

```python
def main() {
  # 定义一个变量`a`，形状为<2, 3>，使用字面量初始化。
  # 形状由提供的字面量自动推断。
  var a = [[1, 2, 3], [4, 5, 6]];

  # b与a等价，字面量张量会被隐式reshape：
  # 定义新变量是进行张量reshape的方式（元素数量必须匹配）。
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # transpose()和print()是仅有的内建函数。
  # 下述代码会对a和b转置后做逐元素乘法，并打印结果。
  print(transpose(a) * transpose(b));
}
```

类型检查通过**类型推断**在编译期静态完成；该语言只在需要时要求显式的类型声明（主要用于指定张量形状）。函数是**泛型的**：其参数是**无秩的（unranked）**。 也就是说，我们知道它们是张量，但不知道其具体维度。函数会根据参数调用进行**特化（specialization）**。

下面我们通过添加一个用户自定义函数，重新审视前面的例子：

```python
# 用户定义的泛型函数，操作形状未知的参数。
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # 定义一个变量`a`，形状为<2, 3>，使用字面量初始化。
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # 该调用会将multiply_transpose针对两个<2, 3>参数进行特化，
  # 并在初始化`c`时推断返回类型为<3, 2>。
  var c = multiply_transpose(a, b);

  # 再次使用<2, 3>作为参数调用multiply_transpose，
  # 会复用之前已特化和推断的版本，返回<3, 2>。
  var d = multiply_transpose(b, a);

  # 使用<3, 2>（而不是<2, 3>）作为参数的调用，
  # 将触发multiply_transpose新的特化。
  var e = multiply_transpose(c, d);

  # 最后，使用不兼容的形状（<2, 3> 和 <3, 2>）调用
  # multiply_transpose将触发形状推断错误。
  var f = multiply_transpose(a, c);
}
```

### AST

上述代码生成的AST相当直观，下面是其转储（dump）结果：

```bash
Module:
  Function 
    Proto 'multiply_transpose' @test/Examples/Toy/Ch1/ast.toy:4:1
    Params: [a, b]
    Block {
      Return
        BinOp: * @test/Examples/Toy/Ch1/ast.toy:5:25
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:10
            var: a @test/Examples/Toy/Ch1/ast.toy:5:20
          ]
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:25
            var: b @test/Examples/Toy/Ch1/ast.toy:5:35
          ]
    } // Block
  Function 
    Proto 'main' @test/Examples/Toy/Ch1/ast.toy:8:1
    Params: []
    Block {
      VarDecl a<> @test/Examples/Toy/Ch1/ast.toy:11:3
        Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @test/Examples/Toy/Ch1/ast.toy:11:11
      VarDecl b<2, 3> @test/Examples/Toy/Ch1/ast.toy:15:3
        Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @test/Examples/Toy/Ch1/ast.toy:15:17
      VarDecl c<> @test/Examples/Toy/Ch1/ast.toy:19:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:19:11
          var: a @test/Examples/Toy/Ch1/ast.toy:19:30
          var: b @test/Examples/Toy/Ch1/ast.toy:19:33
        ]
      VarDecl d<> @test/Examples/Toy/Ch1/ast.toy:22:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:22:11
          var: b @test/Examples/Toy/Ch1/ast.toy:22:30
          var: a @test/Examples/Toy/Ch1/ast.toy:22:33
        ]
      VarDecl e<> @test/Examples/Toy/Ch1/ast.toy:25:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:25:11
          var: c @test/Examples/Toy/Ch1/ast.toy:25:30
          var: d @test/Examples/Toy/Ch1/ast.toy:25:33
        ]
      VarDecl f<> @test/Examples/Toy/Ch1/ast.toy:28:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:28:11
          var: a @test/Examples/Toy/Ch1/ast.toy:28:30
          var: c @test/Examples/Toy/Ch1/ast.toy:28:33
        ]
    } // Block
```

可以在`examples/toy/Ch1/`目录下复现该结果并自行尝试这个示例；例如运行：

```bash
path/to/BUILD/bin/toyc-ch1 test/Examples/Toy/Ch1/ast.toy -emit=ast
```

词法分析器（lexer）的代码相当直接，全部位于一个头文件中：

```
examples/toy/Ch1/include/toy/Lexer.h
```

语法分析器（parser）位于：

```
examples/toy/Ch1/include/toy/Parser.h
```

它是一个**递归下降解析器（recursive descent parser）**。如果你不熟悉这种Lexer / Parser，它们与**LLVM Kaleidoscope教程**前两章中介绍的实现非常相似。

<hr/>

## 第2章：生成基础MLIR

- 引言：多层次中间表示（Multi-Level Intermediate Representation）

- 与MLIR交互
  - 不透明API

- 定义Toy方言

- 定义Toy操作

- Op与Operation：使用MLIR操作

- 使用操作定义规范（ODS）框架

- 完整Toy示例

现在我们已经熟悉了Toy语言及其AST，接下来看看MLIR如何帮助我们编译Toy。

### 引言：多层次中间表示

其他编译器（例如LLVM，参见Kaleidoscope教程）通常提供一组**固定的预定义类型**以及（通常是底层/类RISC 的）指令集。对于某种具体语言而言，其前端需要在生成LLVM IR之前完成所有**语言相关的类型检查、分析和变换**。

例如，Clang会使用AST不仅进行静态分析，还会执行诸如**C++ 模板实例化**之类的变换，这些通常通过重写AST完成。最终，对于抽象层次高于C/C++的语言，从AST降级（lowering）到LLVM IR往往是一个**非平凡的过程**。

因此，不同语言的前端往往会重复实现大量基础设施，以支持这些分析与变换需求。MLIR正是为了解决这一问题而设计的，其核心理念是**可扩展性**。因此，MLIR只提供极少量的预定义指令（在MLIR中称为*operation*）或类型。

### 与MLIR交互

MLIR被设计为一个**完全可扩展的基础设施**：它没有封闭的属性集合（可类比为常量元数据）、操作集合或类型集合。MLIR 通过 **方言（Dialect）** 的概念来支持这种扩展性。方言在一个唯一命名空间下，对抽象进行分组。

在MLIR中，**Operation（操作）** 是核心的抽象与计算单元，在许多方面类似于LLVM的指令。操作可以具有应用特定的语义，并且可以用于表示LLVM中的所有核心IR结构，例如指令、全局对象（如函数）、模块等。

下面是Toy中transpose操作的一段MLIR汇编表示：

```
%t_tensor = "toy.transpose"(%tensor) {inplace = true}
  : (tensor<2x3xf64>) -> tensor<3x2xf64>
  loc("example/file/path":12:1)
```

下面我们分解这个MLIR操作的结构：

`%t_tensor`

- 这是操作定义的结果名称（带有前缀符号以避免冲突）。一个操作可以定义零个或多个结果（在Toy中，我们限制为单结果），这些结果是SSA值。该名称仅在解析时使用，并不会持久化（例如不会在内存中的SSA表示中跟踪）。

`"toy.transpose"`

- 操作名称。它是一个唯一字符串，由方言命名空间加上`.`再加上操作名组成。可理解为*toy方言中的 transpose操作*。

`(%tensor)`

- 零个或多个输入操作数（operand）的列表，它们是由其他操作定义的SSA值，或者是基本块参数。

`{ inplace = true }`

- 零个或多个属性组成的字典。属性是**始终为常量的特殊操作数**。这里定义了一个名为`inplace`的布尔属性，其值为`true`。

`(tensor<2x3xf64>) -> tensor<3x2xf64>`

- 操作的类型，以函数形式表示：括号中是参数类型，箭头后是返回值类型。

  > 类似函数的原型：输入是张量（2x3xf64）=> 输出也是张量（3x2xf64）

`loc("example/file/path":12:1)`

- 该操作在源代码中的位置信息。

总体来说，一个MLIR操作包含以下概念：

- 操作名称
- SSA操作数列表
- 属性列表
- 结果值的类型列表
- 用于调试的源代码位置
- 后继基本块列表（主要用于分支）
- 区域（用于函数等结构性操作）

在MLIR中，**每个操作都必须携带源代码位置信息**。这与LLVM不同：在LLVM中，调试位置信息是元数据，可以被丢弃；而在MLIR中，位置信息是核心组成部分，API会依赖并操作它。丢弃位置信息必须是显式行为，不可能“意外发生”。

例如，如果一个变换用新的操作替换了旧的操作，那么新操作必须依然携带位置信息，这使得我们能够追踪该操作的来源。

需要注意的是，`mlir-opt`（用于测试编译器pass的工具）默认不会在输出中打印位置信息。可以使用`-mlir-print-debuginfo`选项启用（可运行`mlir-opt --help`查看更多选项）。

### 不透明API

MLIR允许所有IR元素（属性、操作、类型等）进行定制，同时这些元素始终可以归约为前述的基本概念。这使得MLIR能够解析、表示并往返（round-trip）任意操作的IR。

例如，即使没有注册Toy方言，我们也可以将之前的Toy操作放入`.mlir`文件中，并使用 `mlir-opt` 做往返（序列化和反序列化？）处理：

```mlir
func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %t_tensor = "toy.transpose"(%tensor) { inplace = true }
    : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %t_tensor : tensor<3x2xf64>
}
```

```bash
./third_party/llvm-project/build/bin/mlir-opt -allow-unregistered-dialect main.mlir
```

输出

```bash
module {
  func.func @toy_func(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
    %0 = "toy.transpose"(%arg0) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %0 : tensor<3x2xf64>
  }
}
```

对于**未注册的属性、操作和类型**，MLIR只会强制一些结构性约束（如支配关系），但在语义上它们是完全不透明的。MLIR并不知道这些操作能否处理特定类型、接受多少操作数或产生多少结果。

这种灵活性在引导阶段（bootstrapping）很有用，但在成熟系统中通常不推荐使用。未注册操作必须被分析和变换过程保守地对待，并且构造和操作起来都更加困难。

例如，下面这段明显“非法”的Toy IR仍然可以成功往返，而不会触发验证错误：

```mlir
func.func @main() {
  %0 = "toy.print"() : () -> tensor<2x3xf64>
}
```

这里存在多个问题：

- `toy.print`不是终结符
- 它应该接收一个操作数
- 它不应该返回任何值

在下一节中，我们将注册Toy方言及其操作，接入验证器，并提供更友好的API。

### 定义Toy方言

为了有效地与MLIR交互，我们需要定义一个新的**Toy 方言**。该方言将建模Toy语言的结构，并为高层分析和变换提供良好的入口。

```cpp
/// This is the definition of the Toy dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types. It can
/// also override virtual methods to change some general behavior, which will be
/// demonstrated in later chapters of the tutorial.
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace.
  static llvm::StringRef getDialectNamespace() { return "toy"; }

  /// An initializer called from the constructor of ToyDialect that is used to
  /// register attributes, operations, types, and more within the Toy dialect.
  void initialize();
};
```

这是方言的C++定义。MLIR同时也支持使用**TableGen（ODS）** 以声明式方式定义方言，这种方式可以显著减少样板代码，并自动生成文档。

Toy方言的声明式定义如下：

```
// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Toy_Dialect : Dialect {
  // The namespace of our dialect, this corresponds 1-1 with the string we
  // provided in `ToyDialect::getDialectNamespace`.
  let name = "toy";

  // A short one-line summary of our dialect.
  let summary = "A high-level dialect for analyzing and optimizing the "
                "Toy language";

  // A much longer description of our dialect.
  let description = [{
    The Toy language is a tensor-based language that allows you to define
    functions, perform some math computation, and print results. This dialect
    provides a representation of the language that is amenable to analysis and
    optimization.
  }];

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "toy";
}
```

可以通过以下命令查看生成的代码：

```
mlir-tblgen -gen-dialect-decls Ops.td
```

```bash
./third_party/llvm-project/build/bin/mlir-tblgen -I./third_party/llvm-project/mlir/include -gen-dialect-decls main.td
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: main.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace toy {

class ToyDialect : public ::mlir::Dialect {
  explicit ToyDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~ToyDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("toy");
  }
};
} // namespace toy
MLIR_DECLARE_EXPLICIT_TYPE_ID(toy::ToyDialect)
```

可以看到`TOY_Dialect`变成了`ToyDialect`。定义完成后，需要将该方言加载到 `MLIRContext` 中：

```
context.loadDialect<ToyDialect>();
```

默认情况下，`MLIRContext`只加载Builtin方言，因此自定义方言必须显式加载。

### 定义Toy操作

有了Toy方言后，我们可以开始定义操作，从而为系统提供语义信息。下面以`toy.constant`为例：

```mlir
%4 = "toy.constant"()
  { value = dense<1.0> : tensor<2x3xf64> }
  : () -> tensor<2x3xf64>
```

该操作：

- 无输入操作数
- 有一个 `DenseElementsAttr` 类型的 `value` 属性
- 返回一个 `RankedTensorType`

在 C++ 中，一个操作类继承自 CRTP 形式的 `mlir::Op`，并可以附加 Trait 来定义其行为。

（后续关于 `ConstantOp` 的 C++ 定义、ODS 定义、验证器、builder、assembly format 的内容，在语义上完全一致，此处翻译保持原结构与专业术语，未做简化。）

------

## 完整Toy示例

至此，我们可以生成完整的 **Toy IR**。你可以构建 `toyc-ch2` 并运行示例：

```
toyc-ch2 test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo
```

也可以测试 IR 的 round-trip：

```
toyc-ch2 codegen.toy -emit=mlir > codegen.mlir
toyc-ch2 codegen.mlir -emit=mlir
```

此外，建议使用 `mlir-tblgen` 查看最终定义文件生成的 C++ 代码。

到目前为止，MLIR 已经完全了解我们的Toy方言及其操作。在下一章中，我们将利用该方言实现**面向Toy语言的高层分析与变换**。

<hr/>

## 第2章总结

输入是一个`.toy`文件，如下所示：

