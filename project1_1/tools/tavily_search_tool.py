import time
from typing import Optional
from langchain.tools import tool
from tavily import TavilyClient
from config.settings import settings
from core.logger import logger
from tools.tool_schemas import SearchQuery


class TavilySearchTool:
    """
    Tavily 搜索工具
    
    集成 Tavily 搜索 API，提供新闻和信息搜索功能
    """
    
    def __init__(self):
        """初始化 Tavily 搜索工具"""
        self.api_key = settings.tavily.api_key
        self.client = TavilyClient(api_key=self.api_key)
        self.logger = logger.bind(module="tavily_search_tool")
    
    def search(self, query: str, max_results: int = 5, search_depth: str = "basic") -> str:
        """
        执行搜索并返回格式化结果
        
        Args:
            query: 搜索关键词或问题
            max_results: 要返回的最大搜索结果数量
            search_depth: 搜索深度 (basic 或 advanced)
            
        Returns:
            格式化的搜索结果
        """
        # 检查搜索关键词是否为空
        if not query:
            return "请提供有效的搜索关键词"
        
        # 限制结果数量范围
        max_results = max(1, min(max_results, 10))
        
        try:
            self.logger.info(f"执行 Tavily 搜索: {query}")
            
            # 调用 Tavily 搜索 API
            response = self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_answer=True,
                include_raw_content=False
            )
            
            # 格式化搜索结果
            return self._format_search_result(response, query)
            
        except Exception as e:
            self.logger.error(f"Tavily 搜索失败: {e}")
            return "搜索服务暂时不可用，请稍后重试"
    
    def _format_search_result(self, response: dict, original_query: str) -> str:
        """
        格式化搜索结果
        
        Args:
            response: Tavily API 返回的响应数据
            original_query: 原始搜索查询
            
        Returns:
            格式化的搜索结果字符串
        """
        result = []
        
        # 添加 AI 生成的答案摘要（如果有）
        if response.get("answer"):
            result.append(f"## 关于 '{original_query}' 的搜索摘要")
            result.append(response["answer"])
            result.append("")
        
        # 添加搜索结果列表
        if response.get("results"):
            result.append(f"## 相关搜索结果 ({len(response['results'])} 条)")
            
            for i, item in enumerate(response["results"], 1):
                title = item.get("title", "无标题")
                url = item.get("url", "")
                content = item.get("content", "")
                
                # 截断过长的内容
                if len(content) > 200:
                    content = content[:200] + "..."
                
                # 格式化单个结果
                result.append(f"{i}. **{title}**")
                if url:
                    result.append(f"   来源: {url}")
                if content:
                    result.append(f"   摘要: {content}")
                result.append("")
        
        # 如果没有结果
        if not result:
            return f"未找到与 '{original_query}' 相关的信息"
        
        # 组合所有结果
        return "\n".join(result)


# 创建 LangChain 工具实例
@tool("tavily_search", args_schema=SearchQuery, return_direct=True)
def tavily_search_tool(query: str, max_results: int = 5, search_depth: str = "basic") -> str:
    """
    使用 Tavily 搜索工具搜索最新信息
    
    可以用于搜索新闻、科技动态、百科知识等各类信息
    
    Args:
        query: 搜索关键词或问题，例如：最新人工智能发展、Python 教程
        max_results: 要返回的最大搜索结果数量，默认值为 5
        search_depth: 搜索深度，可选值：basic（基础搜索）、advanced（高级搜索）
        
    Returns:
        格式化的搜索结果
    """
    search_tool = TavilySearchTool()
    return search_tool.search(query, max_results, search_depth)

