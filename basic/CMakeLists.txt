add_executable(ilist_node_test ilist_node_test.cc)

target_link_libraries(ilist_node_test
  PRIVATE
  LLVMCore
)