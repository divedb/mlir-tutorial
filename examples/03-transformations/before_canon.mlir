// Example with redundant operations that can be canonicalized
func.func @redundant_ops(%x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  
  // x + 0 = x (identity)
  %add_zero = arith.addi %x, %c0 : i32
  
  // x * 1 = x (identity)
  %mul_one = arith.muli %add_zero, %c1 : i32
  
  return %mul_one : i32
}

// Another example with algebraic identities
func.func @algebraic_simplify(%a: i32, %b: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  
  // (a - a) + b = 0 + b = b
  %diff = arith.subi %a, %a : i32
  %result = arith.addi %diff, %b : i32
  
  return %result : i32
}
