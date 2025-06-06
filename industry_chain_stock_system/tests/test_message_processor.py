"""
消息处理模块单元测试
Unit tests for Message Processing Module
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# 添加项目根目录到Python路径 - Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from modules.message_processor import MessageProcessor
from utils.api_client import LLMClient

class TestMessageProcessor(unittest.TestCase):
    """
    消息处理模块测试用例
    Test cases for MessageProcessor module
    """
    
    def setUp(self):
        """测试环境设置 - Test environment setup"""
        # 创建模拟的LLM客户端 - Create mock LLM client
        self.mock_llm_client = MagicMock(spec=LLMClient)
        self.processor = MessageProcessor(self.mock_llm_client)
        
        # 模拟LLM响应 - Mock LLM responses
        self.mock_llm_client.generate_structured.side_effect = self._mock_llm_responses
        
        # 测试数据 - Test data
        self.test_message = "国家发改委发布新能源汽车产业发展规划，目标到2025年新能源汽车销量占比达到25%。"
        
        # 预期的模拟响应 - Expected mock responses
        self.expected_responses = {
            "basic_info": {
                "subject": "新能源汽车产业发展规划",
                "main_entities": ["国家发改委", "新能源汽车"],
                "event_description": "国家发改委发布新能源汽车产业发展规划",
                "time_clues": "无",
                "source_clues": "国家发改委"
            },
            "event_classification": {
                "event_type": "policy",
                "impact_scope": "industry",
                "impact_direction": "positive",
                "time_effect": "long_term",
                "classification_reason": "政府政策规划"
            },
            "impact_analysis": {
                "impact_intensity": "高",
                "affected_industries": ["新能源汽车", "锂电池", "充电桩"],
                "beneficiary_types": ["整车制造商", "电池供应商"],
                "affected_types": [],
                "transmission_mechanism": "政策驱动需求增长",
                "uncertainty_factors": ["政策执行力度", "市场接受度"]
            },
            "entities": {
                "institutions": ["国家发改委"],
                "industries": ["新能源汽车产业"],
                "numerical_info": ["2025年", "25%"]
            },
            "sentiment": {
                "sentiment": "正面",
                "intensity": "强",
                "keywords": ["发展规划", "销量占比"],
                "reason": "政策支持利好产业发展"
            }
        }
        
        self.response_call_count = 0 # 用于模拟不同响应的计数器

    def _mock_llm_responses(self, prompt: str, *args, **kwargs):
        """
        模拟LLM客户端的响应
        Mock responses from LLM client
        
        根据调用次数返回不同的预设响应
        Returns different preset responses based on call count
        """
        self.response_call_count += 1
        
        if "提取基础信息" in prompt:
            return self.expected_responses["basic_info"]
        elif "事件分类" in prompt:
            return self.expected_responses["event_classification"]
        elif "分析其可能的影响" in prompt: # 对应 _analyze_impact
            return self.expected_responses["impact_analysis"]
        elif "提取关键实体" in prompt:
            return self.expected_responses["entities"]
        elif "情感倾向" in prompt:
            return self.expected_responses["sentiment"]
        else:
            # 返回一个通用的空响应或错误，以指示未匹配的提示
            return {"error": "未匹配的提示", "prompt_received": prompt[:100]}

    def test_parse_message_structure(self):
        """测试消息解析结果的基本结构 - Test basic structure of message parsing result"""
        result = self.processor.parse_message(self.test_message)
        
        self.assertIn("original_message", result)
        self.assertIn("basic_info", result)
        self.assertIn("event_classification", result)
        self.assertIn("impact_analysis", result)
        self.assertIn("entities", result)
        self.assertIn("sentiment", result)
        self.assertIn("confidence_score", result)
        
    def test_extract_basic_info(self):
        """测试基础信息提取 - Test basic information extraction"""
        # 重置计数器以确保正确的模拟响应
        self.response_call_count = 0 
        # 直接调用内部方法进行测试
        basic_info = self.processor._extract_basic_info(self.test_message)
        
        self.assertEqual(basic_info["subject"], self.expected_responses["basic_info"]["subject"])
        self.assertListEqual(basic_info["main_entities"], self.expected_responses["basic_info"]["main_entities"])

    def test_classify_event(self):
        """测试事件分类 - Test event classification"""
        self.response_call_count = 0
        event_classification = self.processor._classify_event(self.test_message)
        
        self.assertEqual(event_classification["event_type"], self.expected_responses["event_classification"]["event_type"])
        self.assertEqual(event_classification["impact_direction"], self.expected_responses["event_classification"]["impact_direction"])

    def test_analyze_impact(self):
        """测试影响分析 - Test impact analysis"""
        self.response_call_count = 0
        # 影响分析依赖于事件分类结果，这里我们用模拟的分类结果
        mock_classification = self.expected_responses["event_classification"]
        impact_analysis = self.processor._analyze_impact(self.test_message, mock_classification)
        
        self.assertEqual(impact_analysis["impact_intensity"], self.expected_responses["impact_analysis"]["impact_intensity"])
        self.assertListEqual(impact_analysis["affected_industries"], self.expected_responses["impact_analysis"]["affected_industries"])

    def test_extract_entities(self):
        """测试实体提取 - Test entity extraction"""
        self.response_call_count = 0
        entities = self.processor._extract_entities(self.test_message)
        
        self.assertListEqual(entities["institutions"], self.expected_responses["entities"]["institutions"])
        self.assertListEqual(entities["numerical_info"], self.expected_responses["entities"]["numerical_info"])

    def test_analyze_sentiment(self):
        """测试情感分析 - Test sentiment analysis"""
        self.response_call_count = 0
        sentiment = self.processor._analyze_sentiment(self.test_message)
        
        self.assertEqual(sentiment["sentiment"], self.expected_responses["sentiment"]["sentiment"])
        self.assertEqual(sentiment["intensity"], self.expected_responses["sentiment"]["intensity"])

    def test_confidence_score_calculation(self):
        """测试置信度评分计算 - Test confidence score calculation"""
        # 使用预期的模拟响应来测试置信度计算
        confidence = self.processor._calculate_confidence(
            self.expected_responses["basic_info"],
            self.expected_responses["event_classification"],
            self.expected_responses["impact_analysis"]
        )
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        # 根据模拟数据，期望一个较高的置信度
        self.assertGreater(confidence, 0.7, "置信度应较高，因为模拟数据较完整")

    def test_parse_message_with_llm_failure(self):
        """测试当LLM调用失败时的消息解析 - Test message parsing when LLM call fails"""
        # 配置模拟LLM客户端使其抛出异常
        self.mock_llm_client.generate_structured.side_effect = Exception("LLM API Error")
        
        result = self.processor.parse_message(self.test_message)
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "消息解析失败")
        self.assertIn("error_details", result)

    def test_batch_parse_messages(self):
        """测试批量消息解析 - Test batch message parsing"""
        messages = [self.test_message, "另一条测试消息，关于市场波动。"]
        
        # 确保每次调用 parse_message 时，_mock_llm_responses 的计数器能正确重置或管理
        # 这里简化处理，假设每次 parse_message 内部的 _mock_llm_responses 都能正确工作
        # 理想情况下，_mock_llm_responses 需要更复杂的逻辑来处理批量调用中的多次LLM请求
        
        # 重新配置mock，使其对每次parse_message调用都返回完整的模拟响应序列
        def complex_mock_llm_responses(prompt: str, *args, **kwargs):
            # 这个mock需要根据当前的调用上下文（属于哪个message的哪个步骤）返回数据
            # 为了简化，我们假设它总能返回预期的那组数据
            if "提取基础信息" in prompt: return self.expected_responses["basic_info"]
            if "事件分类" in prompt: return self.expected_responses["event_classification"]
            if "分析其可能的影响" in prompt: return self.expected_responses["impact_analysis"]
            if "提取关键实体" in prompt: return self.expected_responses["entities"]
            if "情感倾向" in prompt: return self.expected_responses["sentiment"]
            return {"error": "未匹配的提示"}

        self.mock_llm_client.generate_structured.side_effect = complex_mock_llm_responses

        results = self.processor.batch_parse_messages(messages)
        
        self.assertEqual(len(results), 2)
        for i, result in enumerate(results):
            self.assertIn("original_message", result)
            self.assertEqual(result["batch_index"], i)
            if "error" not in result: # 检查成功的解析
                self.assertIn("basic_info", result)
                self.assertIn("confidence_score", result)

if __name__ == '__main__':
    unittest.main()
