"""
重试机制工具
使用tenacity库实现指数级退避重试
"""
import logging
from collections.abc import Callable
from typing import Any

import httpx
from requests import RequestException
from tenacity import stop_after_attempt, wait_exponential, before_sleep_log, retry, after_log

logger = logging.getLogger(__name__)


def should_retry_http_error(exception):
    """
        判断HTTP错误是否应该重试
        404错误不应该重试，因为资源不存在
        """
    if isinstance(exception, httpx.HTTPStatusError):
        # 404 Not Found 不应该重试
        if exception.response.status_code == 404:
            return False
        return True
    return True


# 定义需要重试的异常类型
RETRIABLE_EXCEPTIONS = (
    httpx.RequestError,
    httpx.TimeoutException,
    httpx.HTTPStatusError,
    httpx.ConnectError,
    RequestException,
    ConnectionError,
    TimeoutError
)


def create_retry_decorator(
        max_attempts: int = 3,
        min_wait: float = 1.0,
        max_wait: float = 10.0,
        multiplier: float = 2.0,
        exception_types: tuple = RETRIABLE_EXCEPTIONS
):
    """
        创建重试装饰器

        Args:
            max_attempts: 最大重试次数
            min_wait: 最小等待时间（秒）
            max_wait: 最大等待时间（秒）
            multiplier: 指数退避倍数
            exception_types: 需要重试的异常类型
        """

    def custom_retry_condition(exception):
        """自定义重试条件"""
        # 首先检查是否是需要重试的异常类型
        if not isinstance(exception, exception_types):
            return False
        return should_retry_http_error(exception)

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
        retry=custom_retry_condition,
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )


default_retry = create_retry_decorator()

api_retry = create_retry_decorator(
    max_attempts=5,
    min_wait=0.5,
    max_wait=30.0,
    multiplier=2.0,
)


class RetryableHttpClient:
    """
        带重试机制的HTTP客户端
        """

    def __init__(self, base_url: str = "", timeout: float = 30.0) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout, base_url=base_url)

    @api_retry
    async def get(self, url: str, **kwargs) -> httpx.Response:
        """GET请求with重试"""
        logger.info(f"GET请求{url}")
        try:
            response = await self.client.get(url, **kwargs)
            response.raise_for_status()
            logger.info(f"GET请求成功:{url} -> {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"GET请求失败：:{url} -> {str(e)}")
            raise

    @api_retry
    async def post(self, url: str, **kwargs) -> httpx.Response:
        """POST请求with重试"""
        logger.info(f"POST:{url}")
        try:
            response = await self.client.post(url, **kwargs)
            response.raise_for_status()
            logger.info(f"POST:{url} -> {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"POST:{url} -> {str(e)}")
            raise

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


http_client = RetryableHttpClient()


async def retry_async_call(
        func: Callable,
        *args,
        max_attempts: int = 3,
        min_wait: float = 1.0,
        max_wait: float = 10.0,
        **kwargs
) -> Any:
    """
        异步函数重试调用

        Args:
            func: 要重试的异步函数
            max_attempts: 最大重试次数
            min_wait: 最小等待时间
            max_wait: 最大等待时间
            *args, **kwargs: 传递给函数的参数
        """
    retry_decorator = create_retry_decorator(max_attempts=max_attempts, min_wait=min_wait, max_wait=max_wait)

    @retry_decorator
    async def _wrapped_func():
        return await func(*args, **kwargs)

    return await _wrapped_func()

def log_retry_attempt(retry_state):
    logger.warning(f"retry count{retry_state.attempt_number}"
                   f"{retry_state.outcom.exception()}"
                   f"wait {retry_state.next_action.sleep}")
