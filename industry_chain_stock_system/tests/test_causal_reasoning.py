"""
因果推理模块单元测试
Unit tests for Causal Reasoning Module
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# 添加项目根目录到Python路径 - Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from modules.causal_reasoning import CausalReasoningEngine
from utils.api_client import LLMClient

class TestCausalReasoningEngine(unittest.TestCase):
    """
    因果推理引擎测试用例
    Test cases for CausalReasoningEngine module
    """
    
    def setUp(self):
        """测试环境设置 - Test environment setup"""
        self.mock_llm_client = MagicMock(spec=LLMClient)
        self.engine = CausalReasoningEngine(self.mock_llm_client)
        
        # 模拟LLM响应 - Mock LLM responses
        self.mock_llm_client.generate_structured.side_effect = self._mock_llm_responses_for_reasoning
        
        # 测试数据 - Test data
        self.test_event_info = {
            "basic_info": {"event_description": "新能源汽车补贴政策调整"},
            "event_classification": {"event_type_cn": "政策类", "impact_direction": "mixed"},
            "entities": {"companies": ["比亚迪", "宁德时代"]}
        }
        self.test_industry_chain = {
            "industry_name": "新能源汽车",
            "chain_structure": {
                "upstream": [{"segment_name": "锂电池材料"}],
                "midstream": [{"segment_name": "整车制造"}],
                "downstream": [{"segment_name": "销售服务"}]
            },
            "relationships": {} # 简化关系
        }
        
        # 预期的模拟响应 - Expected mock responses
        self.expected_responses = {
            "direct_causality": {
                "directly_affected_segments": [
                    {"segment_name": "整车制造", "impact_mechanism": "补贴调整影响成本"}
                ],
                "affected_company_types": ["新能源车企"],
                "causal_mechanisms": [{"mechanism": "政策调整", "explanation": "..."}]
            },
            "indirect_causality": {
                "second_order_effects": [
                    {"trigger_segment": "整车制造", "affected_segment": "锂电池材料", "confidence": 0.7}
                ],
                "feedback_loops": []
            },
            "key_assumptions": { # 对应 _identify_key_assumptions 的返回
                 "key_assumptions": [
                    {"assumption_type": "市场假设", "assumption_content": "市场需求保持稳定"}
                ]
            }
        }
        self.response_call_count = 0

    def _mock_llm_responses_for_reasoning(self, prompt: str, *args, **kwargs):
        """模拟LLM客户端针对因果推理的响应"""
        self.response_call_count += 1
        if "分析事件的直接因果影响" in prompt:
            return self.expected_responses["direct_causality"]
        elif "分析间接因果关系和传导效应" in prompt:
            return self.expected_responses["indirect_causality"]
        elif "识别关键假设条件" in prompt: # 对应 _identify_key_assumptions
            return self.expected_responses["key_assumptions"]
        # 可以为其他内部LLM调用添加更多模拟分支
        return {"error": "因果推理模块未匹配的提示", "prompt_received": prompt[:100]}

    def test_create_causal_graph_structure(self):
        """测试因果图创建的基本结构 - Test basic structure of causal graph creation"""
        self.response_call_count = 0
        result = self.engine.create_causal_graph(self.test_event_info, self.test_industry_chain)
        
        self.assertNotIn("error", result, f"因果图创建失败: {result.get('error_details')}")
        self.assertIn("event_info", result)
        self.assertIn("direct_causality", result)
        self.assertIn("indirect_causality", result)
        self.assertIn("transmission_paths", result)
        self.assertIn("causal_strength", result)
        self.assertIn("temporal_analysis", result)
        self.assertIn("key_assumptions", result)
        self.assertIn("uncertainty_analysis", result)

    def test_analyze_direct_causality_method(self):
        """测试分析直接因果关系方法 - Test analyze direct causality method"""
        self.response_call_count = 0
        direct_causality = self.engine._analyze_direct_causality(self.test_event_info, self.test_industry_chain)
        
        self.assertTrue(len(direct_causality.get("directly_affected_segments", [])) > 0)
        self.assertTrue(len(direct_causality.get("causal_mechanisms", [])) > 0)

    def test_analyze_indirect_causality_method(self):
        """测试分析间接因果关系方法 - Test analyze indirect causality method"""
        self.response_call_count = 0
        # 模拟直接因果数据
        mock_direct_causality = self.expected_responses["direct_causality"]
        indirect_causality = self.engine._analyze_indirect_causality(mock_direct_causality, self.test_industry_chain)
        
        self.assertTrue(len(indirect_causality.get("second_order_effects", [])) > 0)

    def test_build_transmission_paths(self):
        """测试构建传导路径 - Test building transmission paths"""
        # 使用预设的直接和间接因果数据
        mock_direct = self.expected_responses["direct_causality"]
        mock_indirect = self.expected_responses["indirect_causality"]
        
        paths = self.engine._build_transmission_paths(mock_direct, mock_indirect)
        
        self.assertIsInstance(paths, list)
        self.assertTrue(len(paths) > 0, "应至少构建一条传导路径")
        for path in paths:
            self.assertIn("path_id", path)
            self.assertIn("steps", path)
            self.assertTrue(len(path["steps"]) > 0)

    def test_evaluate_causal_strength(self):
        """测试评估因果强度 - Test evaluating causal strength"""
        # 模拟传导路径数据
        mock_paths = [
            {"path_id": "p1", "confidence": 0.8, "path_length": 1, "path_type": "direct"},
            {"path_id": "p2", "confidence": 0.6, "path_length": 2, "path_type": "indirect"}
        ]
        strength_eval = self.engine._evaluate_causal_strength(mock_paths, self.test_event_info)
        
        self.assertGreater(strength_eval["overall_strength"], 0)
        self.assertEqual(len(strength_eval["path_strengths"]), 2)

    def test_identify_key_assumptions_method(self):
        """测试识别关键假设方法 - Test identifying key assumptions method"""
        self.response_call_count = 0
        mock_paths = [{"path_id": "p1"}] # 简化路径
        assumptions = self.engine._identify_key_assumptions(mock_paths, self.test_event_info)
        
        self.assertIsInstance(assumptions, list)
        self.assertTrue(len(assumptions) > 0, "应识别出至少一个关键假设")
        self.assertIn("assumption_type", assumptions[0])
        self.assertIn("assumption_content", assumptions[0])

    def test_validate_reasoning_chain(self):
        """测试验证推理链 - Test validating reasoning chain"""
        # 模拟一个完整的因果图数据
        mock_causal_graph = {
            "transmission_paths": [{"overall_impact": "positive", "confidence": 0.7}],
            "causal_strength": {"overall_strength": 0.7},
            "temporal_analysis": {"time_windows": {"short_term": {"path_count": 1}}},
            "key_assumptions": [{"importance": "medium"}]
        }
        validation = self.engine.validate_reasoning_chain(mock_causal_graph)
        
        self.assertIn("is_valid", validation)
        self.assertIn("validation_score", validation)
        self.assertGreaterEqual(validation["validation_score"], 0)

    def test_create_causal_graph_with_llm_failure(self):
        """测试当LLM调用失败时的因果图创建 - Test causal graph creation when LLM call fails"""
        self.mock_llm_client.generate_structured.side_effect = Exception("LLM API Error")
        
        result = self.engine.create_causal_graph(self.test_event_info, self.test_industry_chain)
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "因果推理图构建失败")

if __name__ == '__main__':
    unittest.main()
