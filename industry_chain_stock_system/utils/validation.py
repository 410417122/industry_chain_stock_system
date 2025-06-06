"""
验证工具模块
Validation Utility Module

提供数据验证、逻辑校验等功能
Provides data validation, logic verification, etc.
"""

import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# 导入日志工具 - Import logging utility
from .logger import LoggerMixin

class ValidationUtils(LoggerMixin):
    """
    验证工具类
    Validation Utility Class
    """
    
    def __init__(self):
        """初始化验证工具 - Initialize validation utility"""
        self.logger.info("验证工具初始化完成")
    
    def is_valid_stock_code(self, code: str) -> bool:
        """
        验证股票代码是否有效（A股简化版）
        Validate if stock code is valid (A-share simplified version)
        
        Args:
            code: 股票代码 - Stock code
            
        Returns:
            bool: 是否有效 - Whether it's valid
        """
        if not isinstance(code, str):
            return False
        
        # A股代码通常是6位数字 - A-share codes are typically 6 digits
        # 创业板以3开头，科创板以688开头，沪市主板以6开头，深市主板以0开头
        # GEM starts with 3, STAR Market starts with 688, Shanghai main board starts with 6, Shenzhen main board starts with 0
        return bool(re.match(r"^(00[0-2]|300|60[0-35]|688)\d{3}$", code)) or \
               bool(re.match(r"^(000|001|002)\d{3}$", code)) # 补充一些常见开头
    
    def is_valid_industry_name(self, name: str) -> bool:
        """
        验证行业名称是否有效（简化版）
        Validate if industry name is valid (simplified version)
        
        Args:
            name: 行业名称 - Industry name
            
        Returns:
            bool: 是否有效 - Whether it's valid
        """
        if not isinstance(name, str) or len(name.strip()) < 2:
            return False
        
        # 简单检查是否包含常见行业词汇 - Simple check for common industry terms
        common_terms = ["行业", "产业", "制造", "服务", "科技", "金融", "能源", "材料"]
        return any(term in name for term in common_terms)
    
    def validate_event_data(self, event_data: Dict[str, Any]) -> List[str]:
        """
        验证事件数据结构
        Validate event data structure
        
        Args:
            event_data: 事件数据 - Event data
            
        Returns:
            List[str]: 错误信息列表 - List of error messages
        """
        errors = []
        
        # 必需字段检查 - Required fields check
        required_fields = [
            "original_message", "basic_info", "event_classification", 
            "impact_analysis", "entities", "sentiment"
        ]
        for field in required_fields:
            if field not in event_data:
                errors.append(f"事件数据缺少必需字段: {field}")
        
        # 基础信息验证 - Basic info validation
        basic_info = event_data.get("basic_info", {})
        if not basic_info.get("subject"):
            errors.append("事件基础信息缺少'subject'字段")
        if not isinstance(basic_info.get("main_entities", []), list):
            errors.append("'main_entities'字段应为列表")
            
        # 事件分类验证 - Event classification validation
        event_class = event_data.get("event_classification", {})
        if not event_class.get("event_type"):
            errors.append("事件分类缺少'event_type'字段")
        if not event_class.get("impact_direction"):
            errors.append("事件分类缺少'impact_direction'字段")
            
        # 影响分析验证 - Impact analysis validation
        impact_analysis = event_data.get("impact_analysis", {})
        if not impact_analysis.get("impact_intensity"):
            errors.append("影响分析缺少'impact_intensity'字段")
        if not isinstance(impact_analysis.get("affected_industries", []), list):
            errors.append("'affected_industries'字段应为列表")
            
        return errors
    
    def validate_industry_chain_data(self, chain_data: Dict[str, Any]) -> List[str]:
        """
        验证产业链数据结构
        Validate industry chain data structure
        
        Args:
            chain_data: 产业链数据 - Industry chain data
            
        Returns:
            List[str]: 错误信息列表 - List of error messages
        """
        errors = []
        
        if not chain_data.get("industry_name"):
            errors.append("产业链数据缺少'industry_name'字段")
        
        chain_structure = chain_data.get("chain_structure", {})
        for level in ["upstream", "midstream", "downstream"]:
            if level not in chain_structure or not isinstance(chain_structure[level], list):
                errors.append(f"产业链结构缺少或格式错误: {level}")
        
        companies = chain_data.get("companies", [])
        if not isinstance(companies, list):
            errors.append("'companies'字段应为列表")
        elif companies and not all(isinstance(c, dict) and "name" in c for c in companies):
            errors.append("企业列表格式错误，应为包含名称的字典列表")
            
        return errors
        
    def validate_causal_graph_data(self, graph_data: Dict[str, Any]) -> List[str]:
        """
        验证因果图数据结构
        Validate causal graph data structure
        
        Args:
            graph_data: 因果图数据 - Causal graph data
            
        Returns:
            List[str]: 错误信息列表 - List of error messages
        """
        errors = []
        
        if not graph_data.get("event_info"):
            errors.append("因果图数据缺少'event_info'字段")
        
        direct_causality = graph_data.get("direct_causality", {})
        if not isinstance(direct_causality.get("directly_affected_segments", []), list):
            errors.append("'directly_affected_segments'字段应为列表")
            
        transmission_paths = graph_data.get("transmission_paths", [])
        if not isinstance(transmission_paths, list):
            errors.append("'transmission_paths'字段应为列表")
        elif transmission_paths and not all(isinstance(p, dict) and "steps" in p for p in transmission_paths):
            errors.append("传导路径格式错误")
            
        return errors
        
    def validate_recommendation_data(self, recommendations: List[Dict[str, Any]]) -> List[str]:
        """
        验证投资建议数据结构
        Validate investment recommendation data structure
        
        Args:
            recommendations: 投资建议列表 - List of investment recommendations
            
        Returns:
            List[str]: 错误信息列表 - List of error messages
        """
        errors = []
        
        if not isinstance(recommendations, list):
            errors.append("投资建议应为列表格式")
            return errors
            
        for i, rec in enumerate(recommendations):
            if not isinstance(rec, dict):
                errors.append(f"第{i+1}条建议格式错误，应为字典")
                continue
            
            required_fields = ["stock_code", "stock_name", "advice_type", "rationale"]
            for field in required_fields:
                if field not in rec:
                    errors.append(f"第{i+1}条建议缺少必需字段: {field}")
            
            if "stock_code" in rec and not self.is_valid_stock_code(rec["stock_code"]):
                errors.append(f"第{i+1}条建议股票代码无效: {rec['stock_code']}")
        
        return errors

# 便捷函数 - Convenience functions
_validation_utils_instance = None

def get_validation_utils() -> ValidationUtils:
    """获取ValidationUtils单例 - Get ValidationUtils singleton instance"""
    global _validation_utils_instance
    if _validation_utils_instance is None:
        _validation_utils_instance = ValidationUtils()
    return _validation_utils_instance

if __name__ == "__main__":
    # 测试验证工具 - Test validation utility
    validator = get_validation_utils()
    
    # 股票代码验证测试 - Stock code validation test
    print(f"股票代码 '600000' 是否有效: {validator.is_valid_stock_code('600000')}")
    print(f"股票代码 '900000' 是否有效: {validator.is_valid_stock_code('900000')}")
    print(f"股票代码 'ABCDEF' 是否有效: {validator.is_valid_stock_code('ABCDEF')}")
    
    # 行业名称验证测试 - Industry name validation test
    print(f"行业名称 '新能源汽车行业' 是否有效: {validator.is_valid_industry_name('新能源汽车行业')}")
    print(f"行业名称 '餐饮' 是否有效: {validator.is_valid_industry_name('餐饮')}") # 可能为False，取决于具体逻辑
    
    # 事件数据验证测试 - Event data validation test
    valid_event = {
        "original_message": "test", 
        "basic_info": {"subject": "s", "main_entities": ["e"]}, 
        "event_classification": {"event_type": "t", "impact_direction": "d"},
        "impact_analysis": {"impact_intensity": "i", "affected_industries": ["ai"]},
        "entities": {}, 
        "sentiment": {}
    }
    print(f"有效事件数据验证: {validator.validate_event_data(valid_event)}")
    
    invalid_event = {"original_message": "test"}
    print(f"无效事件数据验证: {validator.validate_event_data(invalid_event)}")
    
    # 投资建议验证测试 - Investment recommendation validation test
    valid_recs = [
        {"stock_code": "600000", "stock_name": "浦发银行", "advice_type": "buy", "rationale": "看好"}
    ]
    print(f"有效投资建议验证: {validator.validate_recommendation_data(valid_recs)}")
    
    invalid_recs = [{"code": "000001", "name": "平安银行"}] # 缺少字段
    print(f"无效投资建议验证: {validator.validate_recommendation_data(invalid_recs)}")
    
    print("验证工具测试完成")
