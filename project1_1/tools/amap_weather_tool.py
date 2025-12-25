import requests
import time
from typing import Optional
from langchain.tools import tool
from config.settings import settings
from core.logger import logger
from tools.tool_schemas import WeatherQuery, WeatherResult


# 主要城市的 adcode 映射表（简化版）
# 完整数据可以从 AMap_adcode_citycode.xlsx 文件加载
CITY_ADCODE_MAPPING = {
    "北京": "110000",
    "上海": "310000",
    "广州": "440100",
    "深圳": "440300",
    "杭州": "330100",
    "南京": "320100",
    "武汉": "420100",
    "成都": "510100",
    "重庆": "500000",
    "西安": "610100",
    "长沙": "430100",
    "郑州": "410100",
    "济南": "370100",
    "青岛": "370200",
    "天津": "120000",
    "苏州": "320500",
    "福州": "350100",
    "厦门": "350200",
    "沈阳": "210100",
    "大连": "210200",
    "哈尔滨": "230100",
    "长春": "220100",
    "石家庄": "130100",
    "太原": "140100",
    "合肥": "340100",
    "南昌": "360100",
    "昆明": "530100",
    "贵阳": "520100",
    "兰州": "620100",
    "银川": "640100",
    "西宁": "630100",
    "乌鲁木齐": "650100",
    "拉萨": "540100",
    "南宁": "450100",
    "海口": "460100",
    "呼和浩特": "150100",
    "包头": "150200",
}


class AMapWeatherTool:
    """
    高德地图天气查询工具
    
    封装高德地图天气 API，提供主要城市的天气查询功能
    """
    
    def __init__(self):
        """初始化天气查询工具"""
        self.api_key = settings.amap.api_key
        self.base_url = settings.amap.base_url
        self.logger = logger.bind(module="amap_weather_tool")
    
    def _get_city_adcode(self, city_name: str) -> Optional[str]:
        """
        获取城市的 adcode
        
        Args:
            city_name: 城市名称
            
        Returns:
            城市的 adcode，如果未找到则返回 None
        """
        return CITY_ADCODE_MAPPING.get(city_name)
    
    def _call_amap_weather_api(self, adcode: str, retry_count: int = 3) -> Optional[dict]:
        """
        调用高德地图天气 API
        
        Args:
            adcode: 城市的 adcode
            retry_count: 重试次数
            
        Returns:
            API 响应数据，如果调用失败则返回 None
        """
        url = f"{self.base_url}/weather/weatherInfo"
        params = {
            "key": self.api_key,
            "city": adcode,
            "extensions": "base"  # base: 实时天气, all: 天气预报
        }
        
        for attempt in range(retry_count):
            try:
                self.logger.info(f"调用高德地图天气 API，城市 adcode: {adcode}")
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()  # 检查 HTTP 错误
                
                data = response.json()
                if data.get("status") == "1" and data.get("lives"):
                    return data
                else:
                    self.logger.error(f"高德地图 API 返回错误: {data.get('info', '未知错误')}")
                    return None
            
            except requests.exceptions.RequestException as e:
                self.logger.error(f"高德地图 API 调用失败 (尝试 {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(1)  # 等待 1 秒后重试
            
        return None
    
    def get_weather(self, city_name: str) -> str:
        """
        获取指定城市的天气信息
        
        Args:
            city_name: 城市名称
            
        Returns:
            格式化的天气信息
        """
        # 检查城市名称是否为空
        if not city_name:
            return "请提供有效的城市名称"
        
        # 获取城市的 adcode
        adcode = self._get_city_adcode(city_name)
        if not adcode:
            return f"暂不支持查询城市 '{city_name}' 的天气信息"
        
        # 调用高德地图天气 API
        data = self._call_amap_weather_api(adcode)
        if not data:
            return "天气查询失败，请稍后重试"
        
        # 解析天气信息
        try:
            weather_data = data["lives"][0]
            
            # 使用 Pydantic 模型验证数据
            weather_result = WeatherResult(
                province=weather_data.get("province", ""),
                city=weather_data.get("city", ""),
                weather=weather_data.get("weather", ""),
                temperature=weather_data.get("temperature", ""),
                winddirection=weather_data.get("winddirection", ""),
                windpower=weather_data.get("windpower", ""),
                humidity=weather_data.get("humidity", ""),
                reporttime=weather_data.get("reporttime", ""),
            )
            
            # 格式化输出
            formatted_weather = f"{weather_result.city}今天天气{weather_result.weather}，"
            formatted_weather += f"气温{weather_result.temperature}°C，"
            formatted_weather += f"{weather_result.winddirection}风{weather_result.windpower}级，"
            formatted_weather += f"空气湿度{weather_result.humidity}%，"
            formatted_weather += f"发布时间：{weather_result.reporttime}"
            
            self.logger.info(f"天气查询成功: {city_name} - {weather_result.weather}")
            return formatted_weather
            
        except (KeyError, ValueError) as e:
            self.logger.error(f"解析天气数据失败: {e}")
            return "天气数据解析失败，请稍后重试"


# 创建 LangChain 工具实例
@tool("weather_query", args_schema=WeatherQuery, return_direct=True)
def weather_query_tool(city_name: str) -> str:
    """
    查询指定城市的天气信息
    
    Args:
        city_name: 要查询天气的城市名称，例如：北京、上海、广州
        
    Returns:
        格式化的天气信息
    """
    weather_tool = AMapWeatherTool()
    return weather_tool.get_weather(city_name)

