set(LLVM_TARGET_DEFINITIONS IR/TritonDialect.td)

mlir_tablegen(TritonDialect.h.inc -gen-dialect-decls -dialect=tt)
mlir_tablegen(TritonDialect.cpp.inc -gen-dialect-defs -dialect=tt)
add_mlir_dialect_tablegen_target(MLIRTritonDialectIncGen)

add_custom_target(triton-headers ALL
  DEPENDS MLIRTritonDialectIncGen
)
