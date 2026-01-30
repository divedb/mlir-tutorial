// Function that adds two integers
func.func @add(%arg0: i32, %arg1: i32) -> i32 {
  %result = arith.addi %arg0, %arg1 : i32
  return %result : i32
}

// Function that multiplies two integers
func.func @multiply(%arg0: i32, %arg1: i32) -> i32 {
  %result = arith.muli %arg0, %arg1 : i32
  return %result : i32
}

// Function that demonstrates multiple operations
func.func @compute(%a: i32, %b: i32, %c: i32) -> i32 {
  // Computes (a + b) * c
  %sum = arith.addi %a, %b : i32
  %result = arith.muli %sum, %c : i32
  return %result : i32
}
