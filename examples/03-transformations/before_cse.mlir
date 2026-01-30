// Example with common subexpressions
func.func @duplicate_computation(%a: i32, %b: i32) -> i32 {
  // First computation of sum
  %sum1 = arith.addi %a, %b : i32
  %prod1 = arith.muli %sum1, %sum1 : i32
  
  // Duplicate computation of sum
  %sum2 = arith.addi %a, %b : i32
  %prod2 = arith.muli %sum2, %sum2 : i32
  
  // Result uses both
  %result = arith.addi %prod1, %prod2 : i32
  return %result : i32
}

// More CSE opportunities
func.func @more_duplicates(%x: i32, %y: i32, %z: i32) -> i32 {
  %t1 = arith.muli %x, %y : i32
  %t2 = arith.addi %t1, %z : i32
  
  // Same computation repeated
  %t3 = arith.muli %x, %y : i32
  %t4 = arith.addi %t3, %z : i32
  
  %result = arith.addi %t2, %t4 : i32
  return %result : i32
}
