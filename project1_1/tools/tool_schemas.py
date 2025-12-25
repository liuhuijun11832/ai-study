from pydantic import BaseModel, Field
from typing import List, Optional


class WeatherQuery(BaseModel):
    """
    天气查询工具参数模型
    
    用于定义调用天气查询工具时需要的参数
    
    Example:
        {"city_name": "北京"}
    """
    city_name: str = Field(
        ..., 
        description="要查询天气的城市名称，例如：北京、上海、广州",
        example="北京"
    )


class SearchQuery(BaseModel):
    """
    搜索工具参数模型
    
    用于定义调用搜索工具时需要的参数
    
    Example:
        {"query": "最新人工智能发展", "max_results": 5}
    """
    query: str = Field(
        ..., 
        description="搜索关键词或问题，例如：最新人工智能发展、天气查询 API",
        example="最新人工智能发展"
    )
    max_results: Optional[int] = Field(
        5, 
        description="要返回的最大搜索结果数量，默认值为 5",
        ge=1,  # 至少 1 个结果
        le=10  # 最多 10 个结果
    )
    search_depth: Optional[str] = Field(
        "basic", 
        description="搜索深度，可选值：basic（基础搜索）、advanced（高级搜索）",
        pattern="^(basic|advanced)$"
    )


class WeatherResult(BaseModel):
    """
    天气查询结果模型
    
    用于定义天气查询工具返回的结果结构
    """
    province: str = Field(..., description="省份名称")
    city: str = Field(..., description="城市名称")
    weather: str = Field(..., description="天气现象，例如：晴、阴、雨")
    temperature: str = Field(..., description="实时气温")
    winddirection: str = Field(..., description="风向")
    windpower: str = Field(..., description="风力级别")
    humidity: str = Field(..., description="空气湿度")
    reporttime: str = Field(..., description="发布时间")


class SearchResultItem(BaseModel):
    """
    搜索结果项模型
    
    用于定义搜索工具返回的单个结果结构
    """
    title: str = Field(..., description="结果标题")
    url: str = Field(..., description="结果链接")
    content: str = Field(..., description="内容摘要")
    score: float = Field(..., description="结果相关性评分", ge=0, le=1)


class SearchResult(BaseModel):
    """
    搜索结果模型
    
    用于定义搜索工具返回的完整结果结构
    """
    query: str = Field(..., description="搜索查询")
    answer: Optional[str] = Field(None, description="AI生成的答案摘要")
    results: List[SearchResultItem] = Field(..., description="搜索结果列表")
