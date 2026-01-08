"""
API客户端工具
用于调用FastAPI模拟服务的接口
"""
import logging
from rich.console import Console

from utils.retry import RetryableHttpClient

logger = logging.getLogger(__name__)
console = Console()


class APIClient:
    """
        API客户端类
        负责与FastAPI模拟服务通信
        """

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url
        self.client = RetryableHttpClient(base_url=base_url, timeout=30.0)
