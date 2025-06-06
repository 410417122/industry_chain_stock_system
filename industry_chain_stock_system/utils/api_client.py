"""
API客户端工具
API client utility

提供与大模型API的通信功能
Provides communication functionality with LLM APIs
"""

import json
import time
import asyncio
import re # 导入 re 模块
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import httpx # OpenAI SDK 使用 httpx
from pathlib import Path
from openai import OpenAI, AsyncOpenAI # 导入OpenAI SDK

# 导入配置文件 - Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import API_CONFIG

# 导入日志工具 - Import logging utility
from .logger import get_logger

class LLMClient:
    """
    大语言模型API客户端
    Large Language Model API Client
    
    支持与OpenAI兼容的API进行通信 (使用OpenAI SDK)
    Supports communication with OpenAI-compatible APIs (using OpenAI SDK)
    """
    
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        timeout: int = 30, # 单位：秒
        max_retries: int = 3
    ):
        """
        初始化LLM客户端
        Initialize LLM client
        
        Args:
            base_url: API基础URL - API base URL
            api_key: API密钥 - API key
            model: 默认模型名称 - Default model name
            timeout: 请求超时时间 (秒) - Request timeout in seconds
            max_retries: 最大重试次数 - Maximum retry times
        """
        self.base_url = base_url or API_CONFIG.get("base_url")
        self.api_key = api_key or API_CONFIG.get("api_key")
        self.model = model or API_CONFIG.get("model", "deepseek-chat") # 更新默认模型
        self.timeout = timeout or API_CONFIG.get("timeout", 30) # API_CONFIG中的timeout单位应为秒
        self.max_retries = max_retries
        
        # 设置日志记录器 - Setup logger
        self.logger = get_logger("api_client")
        
        # 初始化OpenAI客户端 - Initialize OpenAI clients
        # httpx.Timeout 用于更精细的超时控制，OpenAI SDK 内部会使用
        sdk_timeout = httpx.Timeout(self.timeout, connect=5.0) # 例如，总超时30秒，连接超时5秒
        
        self.sync_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=sdk_timeout,
            max_retries=self.max_retries # OpenAI SDK v1.0+ 支持直接配置重试次数
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=sdk_timeout,
            max_retries=self.max_retries # OpenAI SDK v1.0+ 支持直接配置重试次数
        )
        
        # 统计信息 - Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        
        self.logger.info(f"LLM客户端初始化完成 - 模型: {self.model}, 基础URL: {self.base_url}, 超时: {self.timeout}s, 重试: {self.max_retries}") # 更新日志信息
    
    def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        system_message: str = None,
        stream: bool = False, # 添加stream参数，与OpenAI SDK一致
        **kwargs
    ) -> str:
        """
        生成文本回复
        Generate text response
        
        Args:
            prompt: 用户提示 - User prompt
            model: 模型名称 - Model name (optional)
            temperature: 温度参数 - Temperature parameter (optional)
            max_tokens: 最大令牌数 - Maximum tokens (optional)
            system_message: 系统消息 - System message (optional)
            stream: 是否流式传输 - Whether to stream (default: False)
            **kwargs: 其他参数 - Other parameters
            
        Returns:
            str: 生成的文本 - Generated text
        """
        # 使用默认值填充参数 - Fill parameters with defaults
        current_model = model or self.model # 使用current_model避免覆盖self.model
        current_temperature = temperature or API_CONFIG.get("temperature", 0.1) # 使用current_temperature
        current_max_tokens = max_tokens or API_CONFIG.get("max_tokens", 2000) # 使用current_max_tokens
        
        # 构建消息列表 - Build message list
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        self.logger.debug(f"发送API请求 - 模型: {current_model}, 提示长度: {len(prompt)}字符, stream: {stream}")
        self.total_requests += 1
        
        try:
            # 使用OpenAI SDK发送请求
            # 注意：OpenAI SDK的max_retries是在客户端初始化时设置的，这里不需要显式循环重试
            completion = self.sync_client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=current_temperature,
                max_tokens=current_max_tokens,
                stream=stream, # 传递stream参数
                **kwargs
            )
            
            if stream:
                # 处理流式响应 (示例：简单拼接)
                # 实际应用中可能需要更复杂的流处理逻辑
                # 此处为了与原方法签名返回str一致，简单拼接
                # 如果需要真正的流式处理，方法签名和调用方式都需要改变
                self.logger.warning("流式响应模式下，当前实现为简单拼接，非真正流式返回。")
                content = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        content += chunk.choices[0].delta.content
                # 流式响应的 token usage 通常在所有 chunk 结束后才完整，或不提供
                # 此处简化，不处理流式响应的 token usage
            else:
                # 处理非流式响应
                content = completion.choices[0].message.content
                if completion.usage:
                    self.total_tokens += completion.usage.total_tokens
                    self.logger.debug(f"令牌使用情况: prompt_tokens={completion.usage.prompt_tokens}, completion_tokens={completion.usage.completion_tokens}, total_tokens={completion.usage.total_tokens}")

            self.successful_requests += 1
            self.logger.debug(f"API请求成功，响应长度: {len(content)}字符")
            return content
            
        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"API请求失败: {e}", exc_info=True) # 添加exc_info=True记录完整堆栈
            raise # 重新抛出异常，让调用者处理或SDK的重试机制处理
    
    # _send_chat_request 方法不再需要，逻辑已合并到 generate 方法中
    
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any] = None,
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成结构化回复
        Generate structured response
        
        Args:
            prompt: 用户提示 - User prompt
            schema: 期望的JSON模式 - Expected JSON schema
            model: 模型名称 - Model name (optional)
            **kwargs: 其他参数 - Other parameters
            
        Returns:
            Dict[str, Any]: 解析后的JSON响应 - Parsed JSON response
        """
        # 在提示中添加JSON格式要求 - Add JSON format requirement to prompt
        json_prompt = f"{prompt}\n\n请以JSON格式返回结果。"
        
        if schema:
            json_prompt += f"\n期望的JSON结构: {json.dumps(schema, ensure_ascii=False, indent=2)}"
        
        # 生成文本回复 - Generate text response
        response_text = self.generate(json_prompt, model=model, **kwargs)
        
        # 尝试解析JSON - Try to parse JSON
        
        # 查找JSON内容 - Find JSON content
        # 优先寻找被```json ... ```包裹的内容
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
        if match:
            json_text = match.group(1).strip()
        else:
            # 如果没有找到markdown格式的JSON，则尝试从头尾查找 '{' 和 '}'
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
            else:
                # 如果还是没有，直接尝试整个文本
                json_text = response_text
        
        try: # 唯一的 try-except 用于 json.loads
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}, 原始响应: {response_text}")
            # 返回包装的错误响应 - Return wrapped error response
            return {
                "error": "JSON解析失败",
                "raw_response": response_text,
                "parse_error": str(e)
            }
    
    async def generate_async(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        system_message: str = None,
        **kwargs
    ) -> str:
        """
        异步生成文本回复
        Generate text response asynchronously
        
        Args:
            prompt: 用户提示 - User prompt
            model: 模型名称 - Model name (optional)
            temperature: 温度参数 - Temperature parameter (optional)
            max_tokens: 最大令牌数 - Maximum tokens (optional)
            system_message: 系统消息 - System message (optional)
            **kwargs: 其他参数 - Other parameters
            
        Returns:
            str: 生成的文本 - Generated text
        """
        # 使用默认值填充参数 - Fill parameters with defaults
        current_model = model or self.model
        current_temperature = temperature or API_CONFIG.get("temperature", 0.1)
        current_max_tokens = max_tokens or API_CONFIG.get("max_tokens", 2000)
        # current_stream = kwargs.pop('stream', False) # 如果要支持 stream，从kwargs中取出
        
        # 构建消息列表 - Build message list
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        self.logger.debug(f"发送异步API请求 - 模型: {current_model}, 提示长度: {len(prompt)}字符")
        self.total_requests += 1

        try:
            # 使用AsyncOpenAI SDK发送异步请求
            completion = await self.async_client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=current_temperature,
                max_tokens=current_max_tokens,
                # stream=current_stream, # 如果要支持 stream
                **kwargs
            )

            # if current_stream:
                # # 处理异步流式响应 (示例：简单拼接)
                # # 注意: 如果启用 stream=True, create() 的返回类型会是 AsyncStream[ChatCompletionChunk]
                # # 需要异步迭代处理，并且最终的 content 和 usage 的获取方式会不同。
                # # 为保持与同步版本 generate() 的返回类型一致 (str)，这里暂时不实现完整的异步流处理。
                # self.logger.warning("异步流式响应模式下，当前实现为简单拼接，非真正流式返回。")
                # content = ""
                # async for chunk in completion: # 注意异步迭代
                    # if chunk.choices[0].delta.content is not None:
                        # content += chunk.choices[0].delta.content
                # # 流式响应的 token usage 通常在所有 chunk 结束后才完整，或不提供
            # else:
            content = completion.choices[0].message.content
            if completion.usage:
                self.total_tokens += completion.usage.total_tokens
                self.logger.debug(f"异步令牌使用情况: prompt_tokens={completion.usage.prompt_tokens}, completion_tokens={completion.usage.completion_tokens}, total_tokens={completion.usage.total_tokens}")
            
            self.successful_requests += 1
            self.logger.debug(f"异步API请求成功，响应长度: {len(content)}字符")
            return content

        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"异步API请求失败: {e}", exc_info=True)
            raise
    
    def batch_generate(
        self,
        prompts: List[str],
        model: str = None,
        concurrent_limit: int = 5,
        **kwargs
    ) -> List[str]:
        """
        批量生成文本回复
        Generate text responses in batch
        
        Args:
            prompts: 提示列表 - List of prompts
            model: 模型名称 - Model name (optional)
            concurrent_limit: 并发限制 - Concurrent limit
            **kwargs: 其他参数 - Other parameters
            
        Returns:
            List[str]: 生成的文本列表 - List of generated texts
        """
        async def run_batch():
            semaphore = asyncio.Semaphore(concurrent_limit)
            
            async def generate_single(prompt):
                async with semaphore:
                    return await self.generate_async(prompt, model=model, **kwargs)
            
            tasks = [generate_single(prompt) for prompt in prompts]
            return await asyncio.gather(*tasks)
        
        # 运行异步批处理 - Run async batch processing
        return asyncio.run(run_batch())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取API使用统计信息
        Get API usage statistics
        
        Returns:
            Dict[str, Any]: 统计信息 - Statistics
        """
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": f"{success_rate:.2f}%",
            "total_tokens": self.total_tokens,
            "average_tokens_per_request": self.total_tokens / self.successful_requests if self.successful_requests > 0 else 0,
            "model": self.model,
            "base_url": self.base_url
        }
    
    def reset_statistics(self):
        """重置统计信息 - Reset statistics"""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.logger.info("API统计信息已重置")
    
    def test_connection(self) -> bool:
        """
        测试API连接
        Test API connection
        
        Returns:
            bool: 连接是否成功 - Whether connection is successful
        """
        try:
            self.logger.info("测试API连接...")
            response = self.generate("你好，请回复'连接成功'", max_tokens=50)
            
            if response and len(response.strip()) > 0:
                self.logger.info(f"API连接测试成功，响应: {response[:100]}...")
                return True
            else:
                self.logger.error("API连接测试失败：响应为空")
                return False
                
        except Exception as e:
            self.logger.error(f"API连接测试失败: {e}")
            return False
    
    def close(self):
        """关闭客户端连接 - Close client connection"""
        # OpenAI SDK v1.x 客户端通常不需要显式关闭HTTP连接，
        # httpx客户端会在其生命周期结束后或垃圾回收时自动处理。
        # 如果需要显式关闭，可以使用 `self.sync_client.close()` 和 `await self.async_client.close()`
        # 但这通常用于确保在程序退出前所有后台任务完成，对于长时间运行的应用可能不需要。
        # 这里我们保持简单，不显式关闭。
        self.logger.info("LLMClient close method called. OpenAI SDK clients manage connections internally.")

class LLMClientPool:
    """
    LLM客户端池
    LLM Client Pool
    
    管理多个LLM客户端实例，提供负载均衡功能
    Manages multiple LLM client instances with load balancing
    """
    
    def __init__(self, clients: List[LLMClient]):
        """
        初始化客户端池
        Initialize client pool
        
        Args:
            clients: LLM客户端列表 - List of LLM clients
        """
        self.clients = clients
        self.current_index = 0
        self.logger = get_logger("llm_client_pool")
        
        self.logger.info(f"LLM客户端池初始化完成，包含 {len(clients)} 个客户端")
    
    def get_next_client(self) -> LLMClient:
        """
        获取下一个可用的客户端（轮询调度）
        Get next available client (round-robin scheduling)
        
        Returns:
            LLMClient: 下一个客户端 - Next client
        """
        client = self.clients[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.clients)
        return client
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        使用池中的客户端生成文本
        Generate text using pool clients
        
        Args:
            prompt: 用户提示 - User prompt
            **kwargs: 其他参数 - Other parameters
            
        Returns:
            str: 生成的文本 - Generated text
        """
        client = self.get_next_client()
        return client.generate(prompt, **kwargs)
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """
        获取池的统计信息
        Get pool statistics
        
        Returns:
            Dict[str, Any]: 池统计信息 - Pool statistics
        """
        total_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "client_count": len(self.clients)
        }
        
        client_stats = []
        for i, client in enumerate(self.clients):
            stats = client.get_statistics()
            client_stats.append({"client_id": i, **stats})
            
            total_stats["total_requests"] += stats["total_requests"]
            total_stats["successful_requests"] += stats["successful_requests"]
            total_stats["failed_requests"] += stats["failed_requests"]
            total_stats["total_tokens"] += stats["total_tokens"]
        
        # 计算总体成功率 - Calculate overall success rate
        if total_stats["total_requests"] > 0:
            total_stats["success_rate"] = f"{total_stats['successful_requests'] / total_stats['total_requests'] * 100:.2f}%"
        else:
            total_stats["success_rate"] = "0.00%"
        
        return {
            "pool_total": total_stats,
            "client_details": client_stats
        }

# 便捷函数 - Convenience functions
def create_default_client() -> LLMClient:
    """
    创建默认LLM客户端
    Create default LLM client
    
    Returns:
        LLMClient: 默认客户端 - Default client
    """
    return LLMClient()

def test_api_connection() -> bool:
    """
    测试默认API连接
    Test default API connection
    
    Returns:
        bool: 连接是否成功 - Whether connection is successful
    """
    client = create_default_client()
    return client.test_connection()

if __name__ == "__main__":
    # 测试API客户端功能 - Test API client functionality
    print("测试LLM API客户端...")
    
    # 创建客户端 - Create client
    # 注意：确保 API_CONFIG 中的 base_url 和 api_key 已正确配置为 DeepSeek 的值
    client = create_default_client() 
    
    # 测试连接 - Test connection
    if client.test_connection(): # test_connection 内部会调用 generate
        print("✓ API连接测试成功")
        
        # 测试文本生成 - Test text generation
        try:
            response = client.generate("请简单介绍一下人工智能", max_tokens=100, model="deepseek-chat") # 明确指定模型
            print(f"✓ 文本生成测试成功: {response[:50]}...")
        except Exception as e:
            print(f"✗ 文本生成测试失败: {e}")
        
        # 测试结构化生成 - Test structured generation
        try:
            structured_response = client.generate_structured(
                "请以JSON格式列出3个AI应用领域，例如：{\"fields\": [\"领域1\", \"领域2\", \"领域3\"]}", # 提示更明确
                schema={"fields": ["field1", "field2", "field3"]}, # schema 仍然可以提供
                model="deepseek-chat" # 明确指定模型
            )
            if "error" not in structured_response:
                print(f"✓ 结构化生成测试成功: {structured_response}")
            else:
                print(f"✗ 结构化生成测试失败: {structured_response}")
        except Exception as e:
            print(f"✗ 结构化生成测试失败: {e}")

        # 测试异步生成
        async def test_async_gen():
            try:
                response_async = await client.generate_async("用异步方式简单介绍一下Python", max_tokens=100, model="deepseek-chat")
                print(f"✓ 异步文本生成测试成功: {response_async[:50]}...")
            except Exception as e:
                print(f"✗ 异步文本生成测试失败: {e}")
        
        asyncio.run(test_async_gen())
        
        # 显示统计信息 - Show statistics
        stats = client.get_statistics()
        print(f"API使用统计: {stats}")
        
    else:
        print("✗ API连接测试失败")
    
    # 关闭客户端 - Close client
    client.close() # 调用 close 方法
    print("测试完成")
