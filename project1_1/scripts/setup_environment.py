#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time


def print_banner():
    """
    打印程序欢迎横幅
    """
    banner = """
    +========================================+
    |  环境设置向导                          |
    |  Environment Setup Wizard             |
    |                                        |
    |  用于配置和验证项目环境                |
    +========================================+
    """
    print(banner)

def check_python_version():
    """
    检查 Python 版本是否符合要求
    
    Returns:
        bool: Python 版本是否符合要求
    """
    print("\n🔍 检查 Python 版本...")
    
    required_major = 3
    required_minor = 9
    
    major, minor, _ = sys.version_info
    
    if major >= required_major and minor >= required_minor:
        print(f"✅ Python 版本符合要求: {sys.version}")
        return True
    else:
        print(f"❌ Python 版本不足: {sys.version}")
        print(f"   需要 Python {required_major}.{required_minor} 或更高版本")
        return False

def check_virtual_environment():
    """
    检查是否在虚拟环境中运行
    
    Returns:
        bool: 是否在虚拟环境中
    """
    print("\n🔍 检查虚拟环境...")
    
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        print(f"✅ 已在虚拟环境中运行: {sys.prefix}")
        return True
    else:
        print("⚠️  未在虚拟环境中运行")
        print("   建议在虚拟环境中运行项目，以避免依赖冲突")
        print("   可以使用以下命令创建和激活虚拟环境：")
        print("   python -m venv venv")
        print("   venv\Scripts\activate  # Windows")
        print("   source venv/bin/activate  # Linux/macOS")
        return False

def check_dependencies():
    """
    检查项目依赖是否已安装
    
    Returns:
        bool: 依赖是否已安装
    """
    print("\n🔍 检查项目依赖...")
    
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"❌ 依赖文件不存在: {requirements_file}")
        return False
    
    try:
        # 使用 pip 检查依赖
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print("✅ 所有依赖已正确安装")
            return True
        else:
            print("⚠️  依赖检查发现问题:")
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            
            print("\n建议重新安装依赖:")
            print(f"   pip install -r {requirements_file}")
            return False
            
    except Exception as e:
        print(f"❌ 检查依赖时发生错误: {e}")
        return False

def install_dependencies():
    """
    安装项目依赖
    
    Returns:
        bool: 依赖是否安装成功
    """
    print("\n📦 安装项目依赖...")
    
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"❌ 依赖文件不存在: {requirements_file}")
        return False
    
    try:
        # 升级 pip
        print("   正在升级 pip...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=True
        )
        
        # 安装依赖
        print(f"   正在安装 {requirements_file}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            check=True
        )
        
        print("✅ 依赖安装成功")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 安装依赖时发生错误: {e}")
        return False

def check_env_file():
    """
    检查环境变量文件是否存在
    
    Returns:
        bool: 环境变量文件是否存在
    """
    print("\n🔍 检查环境变量配置...")
    
    env_file = ".env"
    env_example_file = ".env.example"
    
    if os.path.exists(env_file):
        print(f"✅ 环境变量文件已存在: {env_file}")
        return True
    else:
        print(f"❌ 环境变量文件不存在: {env_file}")
        
        if os.path.exists(env_example_file):
            print(f"   发现示例配置文件: {env_example_file}")
            
            try:
                # 复制示例文件为 .env 文件
                with open(env_example_file, 'r', encoding='utf-8') as f_src:
                    content = f_src.read()
                
                with open(env_file, 'w', encoding='utf-8') as f_dest:
                    f_dest.write(content)
                
                print(f"✅ 已复制示例配置文件为: {env_file}")
                print("   请编辑 .env 文件，填写必要的 API 密钥和配置信息")
                return False  # 虽然创建了文件，但需要用户编辑
                
            except Exception as e:
                print(f"❌ 复制示例配置文件失败: {e}")
                return False
        else:
            print(f"   未发现示例配置文件: {env_example_file}")
            return False

def validate_config():
    """
    验证配置是否有效
    
    Returns:
        bool: 配置是否有效
    """
    print("\n🔍 验证配置...")
    
    try:
        # 添加项目根目录到 Python 路径
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from config.settings import settings
        
        if settings.validate_all():
            print("✅ 配置验证成功")
            return True
        else:
            print("❌ 配置验证失败")
            return False
            
    except ImportError as e:
        print(f"❌ 导入配置模块失败: {e}")
        print("   请确保已正确安装依赖并配置环境变量")
        return False
    except Exception as e:
        print(f"❌ 验证配置时发生错误: {e}")
        return False

def test_api_connections():
    """
    测试 API 连接是否正常
    
    Returns:
        bool: API 连接是否正常
    """
    print("\n🔍 测试 API 连接...")
    
    # 添加项目根目录到 Python 路径
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from config.settings import settings
        
        # 测试 OpenAI API 连接
        print("   测试 OpenAI API 连接...")
        try:
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(
                api_key=settings.openai.api_key,
                base_url=settings.openai.base_url,
                model="gpt-3.5-turbo",
                temperature=0.3
            )
            
            # 测试一个简单的调用
            response = llm.invoke("Hello, world!")
            if response:
                print("✅ OpenAI API 连接正常")
        except Exception as e:
            print(f"❌ OpenAI API 连接失败: {e}")
            
        # 测试高德地图 API 连接
        print("   测试高德地图 API 连接...")
        try:
            import requests
            
            url = f"{settings.amap.base_url}/weather/weatherInfo"
            params = {
                "key": settings.amap.api_key,
                "city": "110000",  # 北京的 adcode
                "extensions": "base"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get("status") == "1":
                print("✅ 高德地图 API 连接正常")
        except Exception as e:
            print(f"❌ 高德地图 API 连接失败: {e}")
            
        # 测试 Tavily API 连接
        print("   测试 Tavily API 连接...")
        try:
            from tavily import TavilyClient
            
            client = TavilyClient(api_key=settings.tavily.api_key)
            
            # 测试一个简单的搜索
            response = client.search(
                query="test",
                search_depth="basic",
                max_results=1
            )
            
            if response:
                print("✅ Tavily API 连接正常")
        except Exception as e:
            print(f"❌ Tavily API 连接失败: {e}")
            
        return True  # 即使某些 API 测试失败，也继续执行
        
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("   请确保已正确安装依赖")
        return False
    except Exception as e:
        print(f"❌ 测试 API 连接时发生错误: {e}")
        return False

def run_application():
    """
    运行应用程序
    """
    print("\n🚀 启动应用程序...")
    
    try:
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\n应用程序已停止")
    except Exception as e:
        print(f"❌ 启动应用程序失败: {e}")

def main():
    """
    程序主入口
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="环境设置向导")
    parser.add_argument("--check-only", action="store_true", help="只检查环境，不进行配置")
    parser.add_argument("--run", action="store_true", help="设置完成后运行应用程序")
    args = parser.parse_args()
    
    # 打印欢迎横幅
    print_banner()
    
    # 检查 Python 版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查是否在虚拟环境中
    check_virtual_environment()
    
    # 检查依赖
    if not args.check_only and not check_dependencies():
        print("\n📦 安装项目依赖...")
        if not install_dependencies():
            sys.exit(1)
    
    # 检查环境变量文件
    check_env_file()
    
    # 验证配置
    if not validate_config():
        print("\n请编辑 .env 文件，确保以下配置项正确设置：")
        print("   - OPENAI_API_KEY: OpenAI API 密钥")
        print("   - AMAP_API_KEY: 高德地图 API 密钥")
        print("   - TAVILY_API_KEY: Tavily 搜索 API 密钥")
        
        # 如果用户没有指定 --check-only，询问是否继续
        if not args.check_only:
            continue_input = input("\n是否继续测试 API 连接？(y/n): ")
            if continue_input.lower() != 'y':
                sys.exit(1)
    
    # 测试 API 连接
    test_api_connections()
    
    # 如果指定了 --run 参数，运行应用程序
    if args.run:
        run_application()
    else:
        print("\n🎉 环境设置向导执行完成")
        print("\n接下来请：")
        print("1. 确保 .env 文件中的 API 密钥已正确配置")
        print("2. 运行 'python main.py' 启动应用程序")
        print("3. 或运行 'python scripts/setup_environment.py --run' 启动应用程序")


if __name__ == "__main__":
    main()