// Constant expressions that can be folded at compile time
func.func @constant_computations() -> i32 {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  
  // (2 + 3) * 4 = 5 * 4 = 20
  %sum = arith.addi %c2, %c3 : i32
  %result = arith.muli %sum, %c4 : i32
  
  return %result : i32
}

// More complex constant folding
func.func @nested_constants() -> i32 {
  %c10 = arith.constant 10 : i32
  %c5 = arith.constant 5 : i32
  %c2 = arith.constant 2 : i32
  
  // ((10 - 5) * 2) = 5 * 2 = 10
  %diff = arith.subi %c10, %c5 : i32
  %result = arith.muli %diff, %c2 : i32
  
  return %result : i32
}

// Constant folding with comparisons
func.func @constant_comparison() -> i1 {
  %c5 = arith.constant 5 : i32
  %c10 = arith.constant 10 : i32
  
  // 5 < 10 = true
  %result = arith.cmpi "slt", %c5, %c10 : i32
  return %result : i1
}
