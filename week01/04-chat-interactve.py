import importlib.util
import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

spec= importlib.util.spec_from_file_location("chat_04", "04-chat.py")
chat_04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat_04)
query = chat_04.query

def interactive_test():
    print("=" * 50)
    print("04-chat.py 交互式测试")
    print("输入'quit'或'exit'退出测试")
    print("=" * 50)
    while True:
        try:
            user_input = input("\n请输入您的问题：").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("bye bye")
                break
            if not user_input:
                print("请输入有效的问题")
                continue
            print("\n正在处理您的问题")
            response = query(user_input)
            print(f"\nAI 响应:")
            print("-" * 30)
            print(response)
            print("-" * 30)
            print(f"响应长度: {len(response)} 字符")
        except KeyboardInterrupt:
            print("\n\n测试被中断")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            print("请重试或输入 'quit' 退出")

if __name__ == "__main__":
    interactive_test()