"""
系统集成测试
System Integration Tests
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
import json

# 添加项目根目录到Python路径 - Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from main import IndustryChainStockSystem
from utils.api_client import LLMClient

class TestSystemIntegration(unittest.TestCase):
    """
    系统集成测试用例
    Test cases for system integration
    """
    
    def setUp(self):
        """测试环境设置 - Test environment setup"""
        # 创建模拟的LLM客户端 - Create mock LLM client
        self.mock_llm_client = MagicMock(spec=LLMClient)
        
        # 模拟LLM响应 - Mock LLM responses
        # 这个mock需要更复杂，能根据不同的调用上下文返回不同的数据
        self.mock_llm_client.generate_structured.side_effect = self._mock_llm_integration_responses
        
        # 预期的模拟响应 - Expected mock responses for a full flow
        self.expected_responses = {
            "message_processor_basic_info": {
                "subject": "集成测试事件", "main_entities": ["测试公司A", "测试行业B"],
                "event_description": "一个用于集成测试的事件", "time_clues": "无", "source_clues": "测试系统"
            },
            "message_processor_classification": {
                "event_type": "market", "impact_scope": "industry", 
                "impact_direction": "positive", "time_effect": "medium_term",
                "classification_reason": "测试驱动"
            },
            "message_processor_impact": {
                "impact_intensity": "中", "affected_industries": ["测试行业B", "相关行业C"],
                "beneficiary_types": ["测试公司A类型"], "affected_types": [],
                "transmission_mechanism": "测试传导", "uncertainty_factors": []
            },
            "message_processor_entities": {"companies": ["测试公司A"]},
            "message_processor_sentiment": {"sentiment": "中性"},
            
            "industry_chain_info": {"definition": "测试行业B是一个..."},
            "industry_chain_companies_llm": {"companies": [{"name": "测试公司A", "code": "TEST01"}]},
            "industry_chain_structure": {"upstream": [{"segment": "上游测试"}]},
            "industry_chain_relationships": {"supply_relationships": [{"from": "上游测试", "to": "测试公司A"}]},
            
            "causal_direct": {"directly_affected_segments": [{"segment_name": "测试公司A"}]},
            "causal_indirect": {"second_order_effects": [{"affected_segment": "相关行业C"}]},
            "causal_assumptions": {"key_assumptions": [{"assumption_content": "市场反应符合预期"}]},
            
            "advisor_relevant_stocks_llm": {"relevant_stocks": [{"name": "测试公司A", "code": "TEST01"}]},
            "advisor_stock_potential": {"potential_impact": "中", "certainty": "中"}
        }
        self.call_counts = {} # 用于跟踪每个模拟响应的调用

    def _mock_llm_integration_responses(self, prompt: str, *args, **kwargs):
        """模拟LLM客户端针对集成测试的响应"""
        # 消息处理模块的模拟
        if "提取基础信息" in prompt:
            return self._get_mock_response("message_processor_basic_info")
        elif "事件分类" in prompt:
            return self._get_mock_response("message_processor_classification")
        elif "分析其可能的影响" in prompt: # MessageProcessor._analyze_impact
            return self._get_mock_response("message_processor_impact")
        elif "提取关键实体" in prompt:
            return self._get_mock_response("message_processor_entities")
        elif "情感倾向" in prompt:
            return self._get_mock_response("message_processor_sentiment")
        
        # 产业链模块的模拟
        elif "行业的主要上市公司" in prompt: # IndustryChainBuilder._get_companies_from_llm
            return self._get_mock_response("industry_chain_companies_llm")
        elif "行业的详细信息" in prompt: # IndustryChainBuilder._get_industry_info
            return self._get_mock_response("industry_chain_info")
        elif "详细产业链结构" in prompt: # IndustryChainBuilder._build_chain_structure
            return self._get_mock_response("industry_chain_structure")
        elif "分析各环节之间的关系" in prompt: # IndustryChainBuilder._analyze_chain_relationships
            return self._get_mock_response("industry_chain_relationships")
            
        # 因果推理模块的模拟
        elif "分析事件的直接因果影响" in prompt: # CausalReasoningEngine._analyze_direct_causality
            return self._get_mock_response("causal_direct")
        elif "分析间接因果关系和传导效应" in prompt: # CausalReasoningEngine._analyze_indirect_causality
            return self._get_mock_response("causal_indirect")
        elif "识别关键假设条件" in prompt: # CausalReasoningEngine._identify_key_assumptions
            return self._get_mock_response("causal_assumptions")
            
        # 投资决策模块的模拟
        elif "筛选出最相关的上市公司" in prompt: # InvestmentAdvisor._filter_relevant_stocks
            return self._get_mock_response("advisor_relevant_stocks_llm")
        elif "评估以下股票在当前事件影响下的投资潜力" in prompt: # InvestmentAdvisor._evaluate_stock_potential
            return self._get_mock_response("advisor_stock_potential")
            
        self.fail(f"集成测试中未匹配的LLM提示: {prompt[:150]}...")
        return {"error": "集成测试未匹配提示"}

    def _get_mock_response(self, key: str):
        """辅助函数，获取并记录模拟响应的调用"""
        self.call_counts[key] = self.call_counts.get(key, 0) + 1
        return self.expected_responses[key]

    @patch('main.LLMClient') # 模拟main.py中实例化的LLMClient
    @patch('modules.industry_chain.ak') # 模拟akshare
    def test_full_analysis_flow_traditional_mode(self, mock_ak, MockLLMClient):
        """测试传统模式下的完整分析流程 - Test full analysis flow in traditional mode"""
        # 配置akshare的模拟返回
        mock_ak.stock_info_a_code_name.return_value.iterrows.return_value = iter([])
        
        # 将setUp中创建的mock_llm_client实例赋给MockLLMClient的返回
        MockLLMClient.return_value = self.mock_llm_client
        
        # 初始化系统，这将使用上面patch的MockLLMClient
        system = IndustryChainStockSystem()
        
        test_message = "这是一个集成测试消息，关于测试行业B和测试公司A。"
        
        # 重置调用计数器
        self.call_counts = {}
        result = system.analyze_message(test_message, mode="standard") # 使用标准模式
        
        self.assertTrue(result.get("success"), f"分析流程失败: {result.get('error')}")
        
        # 验证关键部分是否被调用和生成
        self.assertGreater(self.call_counts.get("message_processor_basic_info", 0), 0, "基础信息提取未调用")
        self.assertIsNotNone(result.get("event_info"), "缺少事件信息")
        
        self.assertGreater(self.call_counts.get("industry_chain_info", 0), 0, "产业链信息获取未调用")
        self.assertIsNotNone(result.get("industry_chain"), "缺少产业链数据")
        
        self.assertGreater(self.call_counts.get("causal_direct", 0), 0, "因果直接分析未调用")
        self.assertIsNotNone(result.get("causal_analysis"), "缺少因果分析数据")
        
        self.assertGreater(self.call_counts.get("advisor_relevant_stocks_llm", 0), 0, "相关股票筛选未调用")
        self.assertIsNotNone(result.get("investment_recommendations"), "缺少投资建议")
        
        # 检查是否有推荐结果
        recommendations = result.get("investment_recommendations", [])
        self.assertTrue(len(recommendations) > 0, "应至少有一条投资建议")
        self.assertIn("stock_name", recommendations[0])
        self.assertEqual(recommendations[0]["stock_name"], "测试公司A")

    @patch('main.LLMClient')
    @patch('agents.crew_manager.Crew') # 模拟CrewAI的Crew类
    @patch('modules.industry_chain.ak')
    def test_full_analysis_flow_agent_mode(self, mock_ak, MockCrew, MockLLMClient):
        """测试Agent模式下的完整分析流程 - Test full analysis flow in agent mode"""
        mock_ak.stock_info_a_code_name.return_value.iterrows.return_value = iter([])
        MockLLMClient.return_value = self.mock_llm_client
        
        # 配置Crew的kickoff模拟返回
        # CrewAI的最终输出是最后一个任务的输出，这里模拟投资建议模块的输出
        mock_crew_instance = MockCrew.return_value
        mock_crew_instance.kickoff.return_value = [
            {
                "stock_code": "TEST01", "stock_name": "测试公司A (Agent)", 
                "advice_type": "buy", "rationale": "Agent模式分析结果",
                "risk_level": "medium"
            }
        ]
        mock_crew_instance.usage_metrics = {"total_tokens": 5000} # 模拟用量

        system = IndustryChainStockSystem()
        test_message = "Agent模式集成测试消息。"
        
        result = system.analyze_message(test_message, mode="agent")
        
        self.assertTrue(result.get("success"), f"Agent模式分析失败: {result.get('error')}")
        
        # 验证Crew是否被调用
        MockCrew.assert_called_once()
        mock_crew_instance.kickoff.assert_called_once()
        
        # 检查结果结构
        self.assertIn("investment_recommendations", result)
        recommendations = result.get("investment_recommendations", [])
        self.assertTrue(len(recommendations) > 0)
        self.assertEqual(recommendations[0]["stock_name"], "测试公司A (Agent)")
        self.assertIn("crew_execution_stats", result)

    def test_interactive_mode_simulation(self):
        """模拟交互式模式的输入和退出 - Simulate interactive mode input and exit"""
        # 这个测试比较难直接进行，因为它涉及到input()的调用
        # 可以考虑使用unittest.mock.patch来模拟input的返回值
        # 或者将交互逻辑分离到更易测试的函数中
        pass

    def test_file_input_mode(self):
        """测试文件输入模式 - Test file input mode"""
        # 这个测试需要实际的文件操作，或者模拟open函数
        # 简单起见，这里仅作占位
        pass

if __name__ == '__main__':
    unittest.main()
