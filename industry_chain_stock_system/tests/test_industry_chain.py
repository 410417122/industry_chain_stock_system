"""
产业链分析模块单元测试
Unit tests for Industry Chain Analysis Module
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# 添加项目根目录到Python路径 - Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from modules.industry_chain import IndustryChainBuilder
from utils.api_client import LLMClient
# 假设AKShare的mock（如果需要）- Assuming mock for AKShare (if needed)
# from unittest.mock import patch

class TestIndustryChainBuilder(unittest.TestCase):
    """
    产业链分析模块测试用例
    Test cases for IndustryChainBuilder module
    """
    
    def setUp(self):
        """测试环境设置 - Test environment setup"""
        self.mock_llm_client = MagicMock(spec=LLMClient)
        self.builder = IndustryChainBuilder(self.mock_llm_client)
        
        # 模拟LLM响应 - Mock LLM responses
        self.mock_llm_client.generate_structured.side_effect = self._mock_llm_responses_for_chain
        
        # 测试行业名称 - Test industry name
        self.test_industry_name = "新能源汽车"
        
        # 预期的模拟响应 - Expected mock responses
        self.expected_responses = {
            "industry_info": {
                "definition": "新能源汽车是指采用新型动力系统...",
                "main_products": ["纯电动汽车", "插电式混合动力汽车"],
                # ... 其他字段 ...
            },
            "companies_llm": { # LLM返回的公司信息
                "companies": [
                    {"name": "比亚迪", "code": "002594", "business": "新能源汽车制造"},
                    {"name": "宁德时代", "code": "300750", "business": "动力电池系统"}
                ]
            },
            "chain_structure": {
                "upstream": [{"segment": "锂电池材料", "description": "..."}],
                "midstream": [{"segment": "整车制造", "description": "..."}],
                "downstream": [{"segment": "销售与服务", "description": "..."}]
            },
            "chain_relationships": {
                "supply_relationships": [
                    {"from": "锂电池材料", "to": "整车制造", "strength": "strong"}
                ],
                # ... 其他关系 ...
            }
        }
        self.response_call_count = 0

    def _mock_llm_responses_for_chain(self, prompt: str, *args, **kwargs):
        """模拟LLM客户端针对产业链构建的响应"""
        self.response_call_count += 1
        if f'"{self.test_industry_name}"行业的主要上市公司' in prompt:
            return self.expected_responses["companies_llm"]
        elif f'"{self.test_industry_name}"行业的详细信息' in prompt:
            return self.expected_responses["industry_info"]
        elif f'构建"{self.test_industry_name}"的详细产业链结构' in prompt:
            return self.expected_responses["chain_structure"]
        elif "分析各环节之间的关系" in prompt: # 对应 _analyze_chain_relationships
            return self.expected_responses["chain_relationships"]
        return {"error": "产业链模块未匹配的提示", "prompt_received": prompt[:100]}

    @patch('modules.industry_chain.ak') # 模拟akshare模块
    def test_build_industry_chain_structure(self, mock_ak):
        """测试产业链构建的基本结构 - Test basic structure of industry chain building"""
        # 配置akshare的模拟返回 - Configure mock return for akshare
        mock_ak.stock_info_a_code_name.return_value = MagicMock() # 返回一个空的DataFrame或模拟数据
        mock_ak.stock_info_a_code_name.return_value.iterrows.return_value = iter([])


        self.response_call_count = 0 # 重置计数器
        result = self.builder.build_industry_chain(self.test_industry_name)
        
        self.assertNotIn("error", result, f"产业链构建失败: {result.get('error_details')}")
        self.assertEqual(result["industry_name"], self.test_industry_name)
        self.assertIn("basic_info", result)
        self.assertIn("companies", result)
        self.assertIn("chain_structure", result)
        self.assertIn("relationships", result)
        self.assertIn("key_nodes", result)

    @patch('modules.industry_chain.ak')
    def test_get_industry_info(self, mock_ak):
        """测试获取行业基础信息 - Test getting basic industry information"""
        self.response_call_count = 0
        info = self.builder._get_industry_info(self.test_industry_name)
        
        self.assertEqual(info["definition"], self.expected_responses["industry_info"]["definition"])

    @patch('modules.industry_chain.ak')
    def test_get_industry_companies(self, mock_ak):
        """测试获取行业公司列表 - Test getting industry company list"""
        mock_ak.stock_info_a_code_name.return_value.iterrows.return_value = iter([])
        self.response_call_count = 0
        
        companies = self.builder._get_industry_companies(self.test_industry_name)
        
        self.assertIsInstance(companies, list)
        # 检查是否至少从LLM获取到公司
        llm_company_names = [c["name"] for c in self.expected_responses["companies_llm"]["companies"]]
        retrieved_names = [c["name"] for c in companies]
        self.assertTrue(any(name in retrieved_names for name in llm_company_names), "未能从LLM获取到预期的公司")


    @patch('modules.industry_chain.ak')
    def test_build_chain_structure_method(self, mock_ak):
        """测试构建产业链结构方法 - Test build chain structure method"""
        self.response_call_count = 0
        # 模拟依赖数据
        mock_industry_info = self.expected_responses["industry_info"]
        mock_companies = self.expected_responses["companies_llm"]["companies"]
        
        structure = self.builder._build_chain_structure(
            self.test_industry_name, mock_industry_info, mock_companies
        )
        
        self.assertIn("upstream", structure)
        self.assertIn("midstream", structure)
        self.assertIn("downstream", structure)
        self.assertTrue(len(structure["upstream"]) > 0, "上游环节不应为空")

    def test_analyze_chain_relationships_method(self):
        """测试分析产业链关系方法 - Test analyze chain relationships method"""
        self.response_call_count = 0
        mock_chain_structure = self.expected_responses["chain_structure"]
        relationships = self.builder._analyze_chain_relationships(mock_chain_structure)
        
        self.assertIn("supply_relationships", relationships)
        self.assertTrue(len(relationships["supply_relationships"]) > 0, "供应关系不应为空")

    def test_identify_key_nodes(self):
        """测试识别关键节点 - Test identifying key nodes"""
        mock_chain_structure = {
            "upstream": [{"segment": "核心材料A", "importance": "high", "barriers": "技术专利"}],
            "midstream": [{"segment": "加工B", "importance": "medium"}],
            "downstream": [{"segment": "应用C", "importance": "low"}]
        }
        mock_relationship_scores = {
            "strong_relationships": ["核心材料A->加工B"]
        }
        
        key_nodes = self.builder._identify_key_nodes(mock_chain_structure, mock_relationship_scores)
        
        self.assertTrue(any(node["name"] == "核心材料A" for node in key_nodes["core_nodes"]))
        self.assertTrue(any(node["name"] == "核心材料A" for node in key_nodes["innovation_nodes"]))
        self.assertTrue(any(node["name"] == "核心材料A" for node in key_nodes["control_points"]))

    @patch('modules.industry_chain.ak')
    def test_build_industry_chain_with_llm_failure(self, mock_ak):
        """测试当LLM调用失败时的产业链构建 - Test industry chain building when LLM call fails"""
        mock_ak.stock_info_a_code_name.return_value.iterrows.return_value = iter([])
        self.mock_llm_client.generate_structured.side_effect = Exception("LLM API Error")
        
        result = self.builder.build_industry_chain(self.test_industry_name)
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "产业链构建失败")

if __name__ == '__main__':
    unittest.main()
