from typing import List, Dict, Optional
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
import time
from config.settings import settings
from core.logger import logger
from tools.amap_weather_tool import weather_query_tool
from tools.tavily_search_tool import tavily_search_tool


class QAAgent:
    """
    LangChain 多任务问答代理
    
    集成多种工具（天气查询、信息搜索等），实现智能对话和工具调用功能
    """
    
    def __init__(self):
        """初始化问答代理"""
        self.logger = logger.bind(module="qa_agent")
        self.memory_store = {}
        
        # 创建 OpenAI 语言模型
        self.llm = ChatOpenAI(
            api_key=settings.openai.api_key,
            base_url=settings.openai.base_url,
            model=settings.openai.model,
            temperature=settings.openai.temperature
        )
        
        # 注册所有工具
        self.tools = [weather_query_tool, tavily_search_tool]
        
        # 创建代理
        self.agent_executor = self._create_agent()
        
        self.logger.info("问答代理初始化完成")
    
    def _create_agent(self) -> AgentExecutor:
        """
        创建 LangChain 代理执行器
        
        Returns:
            配置好的代理执行器
        """
        # 定义系统提示词
        system_prompt = ("你是一个多任务问答助手，你可以帮助用户完成以下任务：\n"
                         "1. 查询天气信息 - 调用天气查询工具\n"
                         "2. 搜索最新信息 - 调用搜索工具\n"
                         "3. 日常对话交流\n"
                         "\n"
                         "请根据用户的问题，智能选择合适的工具来完成任务。\n"
                         "如果无法确定使用哪个工具，请直接询问用户，不要猜测。\n"
                         "请使用友好、自然的语言与用户交流。")
        
        # 创建提示词模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 创建工具代理
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        
        # 创建代理执行器
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True
        )
        
        return agent_executor
    
    def _get_or_create_memory(self, session_id: str) -> ChatMessageHistory:
        """
        获取或创建会话记忆
        
        Args:
            session_id: 会话 ID
            
        Returns:
            会话的消息历史
        """
        if session_id not in self.memory_store:
            self.memory_store[session_id] = ChatMessageHistory()
        
        # 限制会话历史长度
        memory = self.memory_store[session_id]
        if len(memory.messages) > settings.app.max_conversation_history * 2:  # 每条对话有用户和助手两条消息
            memory.messages = memory.messages[-settings.app.max_conversation_history * 2:]
        
        return memory
    
    def chat(self, user_input: str, session_id: str = "default") -> Dict[str, any]:
        """
        处理用户的对话请求
        
        Args:
            user_input: 用户输入
            session_id: 会话 ID，用于区分不同用户的对话历史
            
        Returns:
            包含回复内容、使用的工具和处理时间的字典
        """
        start_time = time.time()
        self.logger.info(f"处理用户查询: {user_input} (会话: {session_id})")
        
        try:
            # 获取会话记忆
            memory = self._get_or_create_memory(session_id)
            
            # 执行代理
            response = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": memory.messages
            })
            
            # 获取回复内容
            reply = response.get("output", "抱歉，我无法回答这个问题")
            
            # 添加到会话历史
            memory.add_user_message(user_input)
            memory.add_ai_message(reply)
            
            # 计算处理时间
            processing_time = round((time.time() - start_time) * 1000, 1)
            
            # 记录使用的工具（如果有）
            used_tools = []
            if "tool_calls" in response:
                for tool_call in response["tool_calls"]:
                    used_tools.append(tool_call["name"])
            
            result = {
                "reply": reply,
                "used_tools": used_tools,
                "processing_time": processing_time,
                "session_id": session_id
            }
            
            self.logger.info(f"用户查询处理完成，耗时 {processing_time}ms，使用工具: {used_tools}")
            return result
            
        except Exception as e:
            self.logger.error(f"处理用户查询失败: {e}")
            return {
                "reply": "抱歉，我暂时无法处理您的请求，请稍后重试",
                "used_tools": [],
                "processing_time": round((time.time() - start_time) * 1000, 1),
                "session_id": session_id
            }
    
    def get_conversation_history(self, session_id: str = "default") -> List[Dict[str, str]]:
        """
        获取会话历史记录
        
        Args:
            session_id: 会话 ID
            
        Returns:
            会话历史记录列表
        """
        memory = self._get_or_create_memory(session_id)
        history = []
        
        for message in memory.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        
        return history
    
    def clear_conversation_history(self, session_id: str = "default") -> bool:
        """
        清除会话历史记录
        
        Args:
            session_id: 会话 ID
            
        Returns:
            是否清除成功
        """
        if session_id in self.memory_store:
            del self.memory_store[session_id]
            self.logger.info(f"清除会话历史: {session_id}")
            return True
        return False


# 创建全局代理实例
qa_agent = QAAgent()


