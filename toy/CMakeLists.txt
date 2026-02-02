add_subdirectory(include/toy)
add_executable(toyc
  toyc.cpp
  mlir/Dialect.cpp
  mlir/MLIRGen.cpp
  parser/AST.cpp
  )
include_directories(include/)
# include_directories(build)
target_link_libraries(toyc
  PRIVATE
    LLVMCore  
    LLVMSupport
    LLVMOption
    MLIRParser
    MLIRSupport
    MLIRCastInterfaces
    MLIRSideEffectInterfaces
    MLIRFuncDialect)
set_target_properties(toyc PROPERTIES
    COMPILE_FLAGS "-fno-rtti"
  )