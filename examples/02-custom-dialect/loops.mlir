// Sum of numbers from 0 to n-1
func.func @sum_range(%n: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  %sum = scf.for %i = %c0 to %n step %c1 
    iter_args(%acc = %c0) -> (index) {
    %next = arith.addi %acc, %i : index
    scf.yield %next : index
  }
  
  return %sum : index
}

// Factorial using while loop
func.func @factorial(%n: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  
  %result:2 = scf.while (%i = %n, %acc = %c1) : (i32, i32) -> (i32, i32) {
    %condition = arith.cmpi "sgt", %i, %c0 : i32
    scf.condition(%condition) %i, %acc : i32, i32
  } do {
  ^bb0(%i: i32, %acc: i32):
    %new_acc = arith.muli %acc, %i : i32
    %c1_block = arith.constant 1 : i32
    %new_i = arith.subi %i, %c1_block : i32
    scf.yield %new_i, %new_acc : i32, i32
  }
  
  return %result#1 : i32
}

// Fibonacci (iterative)
func.func @fibonacci(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  
  // If n <= 1, return n
  %small = arith.cmpi "sle", %n, %c1 : i32
  %result = scf.if %small -> (i32) {
    scf.yield %n : i32
  } else {
    // Calculate using loop
    %fib:3 = scf.while (%i = %c2, %a = %c0, %b = %c1) : (i32, i32, i32) -> (i32, i32, i32) {
      %continue = arith.cmpi "sle", %i, %n : i32
      scf.condition(%continue) %i, %a, %b : i32, i32, i32
    } do {
    ^bb0(%i: i32, %a: i32, %b: i32):
      %next = arith.addi %a, %b : i32
      %i_next = arith.addi %i, %c1 : i32
      scf.yield %i_next, %b, %next : i32, i32, i32
    }
    scf.yield %fib#2 : i32
  }
  
  return %result : i32
}

// Nested loop example
func.func @sum_2d(%rows: index, %cols: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  %sum = scf.for %i = %c0 to %rows step %c1
    iter_args(%outer_sum = %c0) -> (index) {
    %inner_sum = scf.for %j = %c0 to %cols step %c1
      iter_args(%acc = %outer_sum) -> (index) {
      %i_plus_j = arith.addi %i, %j : index
      %next_acc = arith.addi %acc, %i_plus_j : index
      scf.yield %next_acc : index
    }
    scf.yield %inner_sum : index
  }
  
  return %sum : index
}
