from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, ValidationError
import os


class OpenAIConfig(BaseSettings):
    """OpenAI 配置"""
    api_key: str = Field(..., env="OPENAI_API_KEY", description="OpenAI API 密钥")
    base_url: str = Field("https://api.openai.com/v1", env="OPENAI_BASE_URL", description="OpenAI API 基础 URL")
    model: str = Field("gpt-3.5-turbo", env="OPENAI_MODEL", description="OpenAI 模型")
    temperature: float = Field(0.3, env="OPENAI_TEMPERATURE", description="生成文本的随机性")
    max_tokens: int = Field(2048, env="OPENAI_MAX_TOKENS", description="最大生成令牌数")
    
    @field_validator('api_key')
    def validate_api_key(cls, v):
        if not v or not v.strip():
            raise ValueError("OpenAI API 密钥不能为空")
        if not v.startswith("sk-"):
            raise ValueError("OpenAI API 密钥格式不正确，应以 'sk-' 开头")
        return v


class AMapConfig(BaseSettings):
    """高德地图配置"""
    api_key: str = Field(..., env="AMAP_API_KEY", description="高德地图 API 密钥")
    base_url: str = Field("https://restapi.amap.com/v3", env="AMAP_BASE_URL", description="高德地图 API 基础 URL")
    
    @field_validator('api_key')
    def validate_api_key(cls, v):
        if not v or not v.strip():
            raise ValueError("高德地图 API 密钥不能为空")
        # 高德地图API密钥通常是32位字符串
        if len(v.strip()) != 32:
            raise ValueError("高德地图 API 密钥格式不正确，应为32位字符串")
        return v


class TavilyConfig(BaseSettings):
    """Tavily 搜索配置"""
    api_key: str = Field(..., env="TAVILY_API_KEY", description="Tavily 搜索 API 密钥")
    base_url: str = Field("https://api.tavily.com", env="TAVILY_BASE_URL", description="Tavily API 基础 URL")
    
    @field_validator('api_key')
    def validate_api_key(cls, v):
        if not v or not v.strip():
            raise ValueError("Tavily 搜索 API 密钥不能为空")
        if not v.startswith("tvly-"):
            raise ValueError("Tavily 搜索 API 密钥格式不正确，应以 'tvly-' 开头")
        return v


class RedisConfig(BaseSettings):
    """Redis 配置"""
    host: Optional[str] = Field("localhost", env="REDIS_HOST", description="Redis 主机地址")
    port: Optional[int] = Field(6379, env="REDIS_PORT", description="Redis 端口")
    db: Optional[int] = Field(0, env="REDIS_DB", description="Redis 数据库编号")
    password: Optional[str] = Field(None, env="REDIS_PASSWORD", description="Redis 密码")
    
    @field_validator('port')
    def validate_port(cls, v):
        if v is not None:
            if not (1 <= v <= 65535):
                raise ValueError("Redis 端口号必须在 1-65535 之间")
        return v


class AppConfig(BaseSettings):
    """应用配置"""
    name: str = Field("MultiTaskQAAssistant", env="APP_NAME", description="应用名称")
    version: str = Field("1.0.0", env="APP_VERSION", description="应用版本")
    log_level: str = Field("INFO", env="LOG_LEVEL", description="日志级别")
    max_conversation_history: int = Field(50, env="MAX_CONVERSATION_HISTORY", description="最大对话历史记录数")
    cache_ttl: int = Field(3600, env="CACHE_TTL", description="缓存过期时间（秒）")
    max_history_length: int = Field(10, env="MAX_HISTORY_LENGTH", description="最大历史长度")
    retry_max: int = Field(3, env="RETRY_MAX", description="最大重试次数")
    retry_delay: int = Field(2, env="RETRY_DELAY", description="重试延迟（秒）")
    
    @field_validator('log_level')
    def validate_log_level(cls, v):
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_log_levels:
            raise ValueError(f"无效的日志级别: {v}，有效级别为: {', '.join(valid_log_levels)}")
        return v.upper()


class Settings(BaseSettings):
    """项目配置管理"""
    # 统一配置所有嵌套模型，不需要单独初始化
    openai: OpenAIConfig
    amap: AMapConfig
    tavily: TavilyConfig
    redis: RedisConfig
    app: AppConfig
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
    )
    
    def validate_all(self) -> bool:
        """验证所有配置项是否有效"""
        try:
            # 验证所有子配置
            # Pydantic 会自动验证子模型，所以这里只需要检查是否成功初始化
            print("所有配置验证通过")
            return True
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False


# 创建全局配置实例
settings = Settings()


