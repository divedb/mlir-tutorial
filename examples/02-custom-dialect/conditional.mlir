// Absolute value using conditional
func.func @abs(%x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %is_negative = arith.cmpi "slt", %x, %c0 : i32
  %result = scf.if %is_negative -> (i32) {
    %neg = arith.subi %c0, %x : i32
    scf.yield %neg : i32
  } else {
    scf.yield %x : i32
  }
  return %result : i32
}

// Maximum of two numbers
func.func @max(%a: i32, %b: i32) -> i32 {
  %cmp = arith.cmpi "sgt", %a, %b : i32
  %result = arith.select %cmp, %a, %b : i32
  return %result : i32
}

// Sign function: returns -1, 0, or 1
func.func @sign(%x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cn1 = arith.constant -1 : i32
  
  %is_positive = arith.cmpi "sgt", %x, %c0 : i32
  %is_negative = arith.cmpi "slt", %x, %c0 : i32
  
  %result = scf.if %is_positive -> (i32) {
    scf.yield %c1 : i32
  } else {
    %inner_result = scf.if %is_negative -> (i32) {
      scf.yield %cn1 : i32
    } else {
      scf.yield %c0 : i32
    }
    scf.yield %inner_result : i32
  }
  return %result : i32
}
