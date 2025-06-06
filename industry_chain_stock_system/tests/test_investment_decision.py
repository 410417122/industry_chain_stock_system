"""
投资决策模块单元测试
Unit tests for Investment Decision Module
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# 添加项目根目录到Python路径 - Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from modules.investment_decision import InvestmentAdvisor
from utils.api_client import LLMClient
# 假设AKShare的mock - Assuming mock for AKShare
# from unittest.mock import patch 

class TestInvestmentAdvisor(unittest.TestCase):
    """
    投资顾问模块测试用例
    Test cases for InvestmentAdvisor module
    """
    
    def setUp(self):
        """测试环境设置 - Test environment setup"""
        self.mock_llm_client = MagicMock(spec=LLMClient)
        self.advisor = InvestmentAdvisor(self.mock_llm_client)
        
        # 模拟LLM响应 - Mock LLM responses
        self.mock_llm_client.generate_structured.side_effect = self._mock_llm_responses_for_advisor
        
        # 测试数据 - Test data
        self.test_causal_graph = {
            "event_info": {"event_classification": {"impact_direction": "positive"}},
            "direct_causality": {
                "directly_affected_segments": [{"segment_name": "充电桩制造"}],
                "affected_company_types": ["充电桩制造商", "新能源车企"]
            },
            "transmission_paths": [{"path_id": "p1", "confidence": 0.8}],
            "causal_strength": {"overall_strength": 0.7},
            "uncertainty_analysis": {"overall_uncertainty": 0.3}
        }
        self.test_industry_chain = {
            "industry_name": "新能源汽车",
            "companies": [
                {"name": "特锐德", "code": "300001", "business": "充电桩制造", "position": "下游", "market_status": "充电桩制造商"},
                {"name": "比亚迪", "code": "002594", "business": "新能源汽车", "position": "中游", "market_status": "新能源车企"}
            ]
        }
        
        # 预期的模拟响应 - Expected mock responses
        self.expected_responses = {
            "relevant_stocks_llm": { # LLM筛选相关股票的返回
                "relevant_stocks": [
                    {"name": "特锐德", "code": "300001", "relevance_reason": "直接受益于充电桩需求"},
                    {"name": "比亚迪", "code": "002594", "relevance_reason": "新能源车市场领导者"}
                ]
            },
            "stock_potential_teruide": { # 特锐德潜力评估
                "potential_impact": "高", "time_horizon": "中期", "certainty": "高",
                "bullish_reasons": ["政策利好充电桩"], "risk_factors": ["竞争加剧"]
            },
            "stock_potential_byd": { # 比亚迪潜力评估
                "potential_impact": "中", "time_horizon": "长期", "certainty": "中",
                "bullish_reasons": ["市场份额提升"], "risk_factors": ["原材料价格波动"]
            }
        }
        self.response_call_count = 0

    def _mock_llm_responses_for_advisor(self, prompt: str, *args, **kwargs):
        """模拟LLM客户端针对投资决策的响应"""
        self.response_call_count += 1
        if "筛选出最相关的上市公司" in prompt: # 对应 _filter_relevant_stocks
            return self.expected_responses["relevant_stocks_llm"]
        elif "评估以下股票在当前事件影响下的投资潜力" in prompt:
            if "特锐德" in prompt:
                return self.expected_responses["stock_potential_teruide"]
            elif "比亚迪" in prompt:
                return self.expected_responses["stock_potential_byd"]
        return {"error": "投资决策模块未匹配的提示", "prompt_received": prompt[:100]}

    @patch('modules.investment_decision.ak') # 模拟akshare
    def test_generate_recommendations_structure(self, mock_ak):
        """测试投资建议生成的基本结构 - Test basic structure of recommendation generation"""
        # 配置akshare的模拟返回 (如果generate_recommendations内部直接或间接调用了akshare)
        # mock_ak.some_function.return_value = ...
        self.response_call_count = 0

        recommendations = self.advisor.generate_recommendations(
            self.test_causal_graph, self.test_industry_chain, top_n=2
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), 2, "应生成两条建议")
        for rec in recommendations:
            self.assertIn("stock_code", rec)
            self.assertIn("stock_name", rec)
            self.assertIn("advice_type", rec)
            self.assertIn("rationale", rec)
            self.assertIn("risk_level", rec)

    def test_extract_key_impacts(self):
        """测试提取关键影响信息 - Test extracting key impact information"""
        impacts = self.advisor._extract_key_impacts(self.test_causal_graph)
        
        self.assertIn("direct_impacts", impacts)
        self.assertIn("strong_transmission_paths", impacts)
        self.assertEqual(impacts["overall_impact_direction"], "positive")

    @patch('modules.investment_decision.ak')
    def test_filter_relevant_stocks_method(self, mock_ak):
        """测试筛选相关股票方法 - Test filtering relevant stocks method"""
        self.response_call_count = 0
        # 模拟关键影响数据
        mock_key_impacts = {
            "affected_company_types": ["充电桩制造商", "新能源车企"],
            "direct_impacts": [{"segment_name": "充电桩制造"}]
        }
        
        stocks = self.advisor._filter_relevant_stocks(mock_key_impacts, self.test_industry_chain)
        
        self.assertIsInstance(stocks, list)
        self.assertTrue(len(stocks) > 0, "应筛选出相关股票")
        # 检查是否包含预期的公司
        stock_names = [s["name"] for s in stocks]
        self.assertIn("特锐德", stock_names)
        self.assertIn("比亚迪", stock_names)

    def test_evaluate_stock_potential_method(self):
        """测试评估股票潜力方法 - Test evaluating stock potential method"""
        self.response_call_count = 0
        # 模拟相关股票和关键影响
        mock_relevant_stocks = [
            {"name": "特锐德", "code": "300001", "business": "充电桩"},
            {"name": "比亚迪", "code": "002594", "business": "新能源车"}
        ]
        mock_key_impacts = self.advisor._extract_key_impacts(self.test_causal_graph) # 使用之前的方法生成
        
        potentials = self.advisor._evaluate_stock_potential(
            mock_relevant_stocks, mock_key_impacts, self.test_causal_graph
        )
        
        self.assertEqual(len(potentials), 2)
        for p in potentials:
            self.assertIn("stock_name", p)
            self.assertIn("potential_impact", p)
            self.assertIn("bullish_reasons", p)

    def test_formulate_investment_advice(self):
        """测试制定投资建议 - Test formulating investment advice"""
        # 模拟股票潜力数据
        mock_potentials = [
            {"stock_name": "特锐德", "potential_impact": "高", "certainty": "高", "bullish_reasons": ["政策利好"]},
            {"stock_name": "比亚迪", "potential_impact": "中", "certainty": "中", "bullish_reasons": ["市场份额"]},
            {"stock_name": "公司C", "potential_impact": "低", "certainty": "低"}
        ]
        
        advice = self.advisor._formulate_investment_advice(mock_potentials, top_n=2)
        
        self.assertEqual(len(advice), 2)
        self.assertEqual(advice[0]["stock_name"], "特锐德") # 基于排序逻辑
        self.assertEqual(advice[0]["advice_type"], "buy")

    def test_assess_recommendation_risk(self):
        """测试评估建议风险 - Test assessing recommendation risk"""
        # 模拟建议数据
        mock_recommendations = [
            {"certainty": "高"},
            {"certainty": "低"}
        ]
        
        recommendations_with_risk = self.advisor._assess_recommendation_risk(
            mock_recommendations, self.test_causal_graph
        )
        
        self.assertEqual(len(recommendations_with_risk), 2)
        self.assertIn("risk_level", recommendations_with_risk[0])
        self.assertEqual(recommendations_with_risk[0]["risk_level"], "low") # 高确定性 -> 低风险
        self.assertEqual(recommendations_with_risk[1]["risk_level"], "high") # 低确定性 -> 高风险

    @patch('modules.investment_decision.ak')
    def test_generate_recommendations_with_llm_failure(self, mock_ak):
        """测试当LLM调用失败时的建议生成 - Test recommendation generation when LLM call fails"""
        self.mock_llm_client.generate_structured.side_effect = Exception("LLM API Error")
        
        recommendations = self.advisor.generate_recommendations(
            self.test_causal_graph, self.test_industry_chain
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), 0, "LLM失败时不应生成建议")

if __name__ == '__main__':
    unittest.main()
