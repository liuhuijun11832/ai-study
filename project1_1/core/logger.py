from loguru import logger
import os
from datetime import datetime
from config.settings import settings


class Logger:
    """日志系统实现"""
    
    def __init__(self):
        """初始化日志系统"""
        # 创建日志目录
        os.makedirs("logs", exist_ok=True)
        
        # 获取当前日期
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # 移除默认的控制台日志配置
        logger.remove()
        
        # 配置控制台日志
        logger.add(
            sink=lambda msg: print(msg, end=""),  # 控制台输出
            level=settings.app.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )
        
        # 配置应用日志文件
        logger.add(
            sink=f"logs/app_{current_date}.log",
            level=settings.app.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="1 day",  # 每天轮转
            retention="7 days",  # 保留7天
            compression="zip",  # 压缩归档
        )
        
        # 配置错误日志文件
        logger.add(
            sink=f"logs/error_{current_date}.log",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="1 day",
            retention="14 days",
            compression="zip",
        )
        
        # 配置 API 调用日志文件
        logger.add(
            sink=f"logs/api_{current_date}.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[module]} | {message}",
            rotation="1 day",
            retention="7 days",
            compression="zip",
        )
        
        # 记录初始化信息
        logger.info("日志系统初始化完成")
        logger.info(f"应用名称: {settings.app.name}, 版本: {settings.app.version}")
        logger.info(f"日志级别: {settings.app.log_level}")
    
    @staticmethod
    def get_logger(module_name: str = "app"):
        """获取日志记录器"""
        return logger.bind(module=module_name)


# 创建全局日志实例
logger = Logger.get_logger()


