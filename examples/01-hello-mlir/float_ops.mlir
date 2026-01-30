// Floating point operations
func.func @distance_squared(%x1: f32, %y1: f32, %x2: f32, %y2: f32) -> f32 {
  // Computes (x2 - x1)^2 + (y2 - y1)^2
  %dx = arith.subf %x2, %x1 : f32
  %dy = arith.subf %y2, %y1 : f32
  %dx2 = arith.mulf %dx, %dx : f32
  %dy2 = arith.mulf %dy, %dy : f32
  %dist2 = arith.addf %dx2, %dy2 : f32
  return %dist2 : f32
}

// Average of three numbers
func.func @average(%a: f32, %b: f32, %c: f32) -> f32 {
  %sum_ab = arith.addf %a, %b : f32
  %sum_abc = arith.addf %sum_ab, %c : f32
  %divisor = arith.constant 3.0 : f32
  %avg = arith.divf %sum_abc, %divisor : f32
  return %avg : f32
}

// Polynomial: a*x^2 + b*x + c
func.func @polynomial(%x: f32, %a: f32, %b: f32, %c: f32) -> f32 {
  %x2 = arith.mulf %x, %x : f32
  %ax2 = arith.mulf %a, %x2 : f32
  %bx = arith.mulf %b, %x : f32
  %ax2_bx = arith.addf %ax2, %bx : f32
  %result = arith.addf %ax2_bx, %c : f32
  return %result : f32
}
