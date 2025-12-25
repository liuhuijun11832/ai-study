#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import readline
from typing import Optional
from config.settings import settings, ValidationError
from core.logger import logger
from agents.qa_agent import qa_agent

def print_banner():
    """
    打印程序欢迎横幅
    """
    banner = """
    +========================================+
    |  LangChain 多任务问答助手              |
    |  Multi-Task QA Assistant               |
    |                                        |
    |  支持功能：                            |
    |  ✅ 天气查询 - 例如：查询北京天气      |
    |  ✅ 信息搜索 - 例如：搜索最新科技新闻  |
    |  ✅ 日常对话交流                       |
    |                                        |
    |  输入 'exit' 或 'quit' 退出程序        |
    |  输入 'clear' 清空对话历史             |
    +========================================+
    """
    print(banner)

def validate_config():
    """
    验证配置是否有效
    
    Returns:
        bool: 配置是否有效
    """
    try:
        settings.validate_all()
        return True
    except ValidationError as e:
        logger.error(f"配置验证失败: {e}")
        return False
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return False

def handle_user_input(user_input: str, session_id: str) -> bool:
    """
    处理用户输入
    
    Args:
        user_input: 用户输入的内容
        session_id: 会话 ID
        
    Returns:
        bool: 是否继续运行程序
    """
    # 去除输入两端的空格
    user_input = user_input.strip()
    
    # 检查退出命令
    if user_input.lower() in ["exit", "quit", "退出", "结束"]:
        print("\n感谢使用多任务问答助手，再见！")
        return False
    
    # 检查清空命令
    if user_input.lower() in ["clear", "清空", "清除历史"]:
        qa_agent.clear_conversation_history(session_id)
        print("\n对话历史已清空")
        return True
    
    # 检查帮助命令
    if user_input.lower() in ["help", "帮助", "?", "？"]:
        print("\n支持的功能：")
        print("  - 天气查询：例如 '查询北京天气'")
        print("  - 信息搜索：例如 '搜索最新人工智能发展'")
        print("  - 日常对话：直接输入问题或内容")
        print("  - 退出程序：输入 'exit' 或 'quit'")
        print("  - 清空历史：输入 'clear'")
        return True
    
    # 检查空输入
    if not user_input:
        return True
    
    # 处理正常输入
    try:
        # 调用问答代理
        response = qa_agent.chat(user_input, session_id)
        
        # 打印回复
        print("\n" + "=" * 50)
        print(response["reply"])
        
        # 打印使用的工具和处理时间
        if response["used_tools"]:
            print("\n🔧 使用工具:", ", ".join(response["used_tools"]))
        print(f"⏱️  处理时间: {response['processing_time']}ms")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"处理用户输入时发生错误: {e}")
        print("\n抱歉，处理您的请求时发生了错误，请稍后重试")
    
    return True

def main():
    """
    程序主入口
    """
    # 打印欢迎横幅
    print_banner()
    
    # 验证配置
    logger.info("开始验证配置...")
    if not validate_config():
        print("\n❌ 配置验证失败，请检查 .env 文件中的配置项")
        print("请确保已正确设置以下环境变量：")
        print("  - OPENAI_API_KEY: OpenAI API 密钥")
        print("  - AMAP_API_KEY: 高德地图 API 密钥")
        print("  - TAVILY_API_KEY: Tavily 搜索 API 密钥")
        print("\n您可以复制 .env.example 文件为 .env 并填写相应的配置项")
        sys.exit(1)
    
    print("\n✅ 配置验证成功，正在初始化...")
    
    # 设置会话 ID（可以根据需要生成唯一的会话 ID）
    session_id = "user_default"
    
    print("\n您好！我是多任务问答助手，我可以帮您查询天气、搜索信息，或者和您聊天。")
    print("请输入您的问题或需求：")
    
    # 主循环
    try:
        while True:
            try:
                # 获取用户输入
                user_input = input("\n您: ")
                
                # 处理用户输入
                if not handle_user_input(user_input, session_id):
                    break
                    
            except KeyboardInterrupt:
                print("\n\n感谢使用多任务问答助手，再见！")
                break
                
    except Exception as e:
        logger.error(f"程序运行时发生未处理的错误: {e}")
        print("\n❌ 程序发生严重错误，请查看日志文件了解详情")
        sys.exit(1)


if __name__ == "__main__":
    main()