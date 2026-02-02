#include <iostream>

#include "llvm/ADT/ilist.h"
#include "llvm/ADT/simple_ilist.h"

struct MyNode : public llvm::ilist_node<MyNode> {
  int Value;

  MyNode(int V) : Value(V) {}

  ~MyNode() {
    std::cout << "Destroying MyNode with value: " << Value << std::endl;
  }
};

void BasicUsage() {
  std::cout << "=== 基本用法示例 ===" << std::endl;

  // 创建一个简单链表
  llvm::simple_ilist<MyNode> list;

  // 创建一些节点
  MyNode node1(1);
  MyNode node2(2);
  MyNode node3(3);

  // 将节点添加到链表
  list.push_back(node1);
  list.push_back(node2);
  list.push_front(node3);

  // 遍历链表
  std::cout << "链表内容: ";
  for (const auto& node : list) {
    std::cout << node.Value << " ";
  }
  std::cout << std::endl;

  // 从链表中移除节点
  list.remove(node2);

  std::cout << "移除节点2后: ";
  for (const auto& node : list) {
    std::cout << node.Value << " ";
  }
  std::cout << std::endl;

  // 检查链表大小
  std::cout << "链表大小: " << list.size() << std::endl;
}

int main() { BasicUsage(); }