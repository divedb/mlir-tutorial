// A simple function that returns a constant
func.func @return_constant() -> i32 {
  %c42 = arith.constant 42 : i32
  return %c42 : i32
}
