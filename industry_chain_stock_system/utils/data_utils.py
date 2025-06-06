"""
数据处理工具模块
Data Processing Utility Module

提供数据清洗、转换、验证等常用功能
Provides common functions for data cleaning, transformation, validation, etc.
"""

import json
import re
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union
import pandas as pd

# 导入日志工具 - Import logging utility
from .logger import LoggerMixin

class DataUtils(LoggerMixin):
    """
    数据处理工具类
    Data Processing Utility Class
    """
    
    def __init__(self):
        """初始化数据处理工具 - Initialize data processing utility"""
        self.logger.info("数据处理工具初始化完成")
    
    def clean_text(self, text: str) -> str:
        """
        清理文本数据
        Clean text data
        
        Args:
            text: 原始文本 - Original text
            
        Returns:
            str: 清理后的文本 - Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # 移除多余空格 - Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 移除特殊字符（可选）- Remove special characters (optional)
        # text = re.sub(r'[^\w\s.,;:?!()-]', '', text)
        
        return text
    
    def normalize_date(self, date_str: str, fmt: str = "%Y-%m-%d") -> Optional[str]:
        """
        规范化日期字符串
        Normalize date string
        
        Args:
            date_str: 日期字符串 - Date string
            fmt: 输出格式 - Output format
            
        Returns:
            Optional[str]: 规范化后的日期字符串或None - Normalized date string or None
        """
        if not date_str:
            return None
        
        # 尝试多种常见日期格式 - Try multiple common date formats
        common_formats = [
            "%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日",
            "%Y.%m.%d", "%m/%d/%Y", "%d/%m/%Y"
        ]
        
        parsed_date = None
        for input_fmt in common_formats:
            try:
                parsed_date = datetime.strptime(date_str, input_fmt)
                break
            except ValueError:
                continue
        
        if parsed_date:
            return parsed_date.strftime(fmt)
        else:
            self.logger.warning(f"无法解析日期字符串: {date_str}")
            return None
    
    def extract_numbers(self, text: str) -> List[float]:
        """
        从文本中提取数字
        Extract numbers from text
        
        Args:
            text: 文本 - Text
            
        Returns:
            List[float]: 提取的数字列表 - List of extracted numbers
        """
        if not isinstance(text, str):
            return []
        
        # 匹配整数和浮点数 - Match integers and floating-point numbers
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        return [float(num) for num in numbers]
    
    def dict_to_dataframe(self, data: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """
        将字典列表转换为Pandas DataFrame
        Convert list of dictionaries to Pandas DataFrame
        
        Args:
            data: 字典列表 - List of dictionaries
            
        Returns:
            Optional[pd.DataFrame]: DataFrame或None - DataFrame or None
        """
        if not data or not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            self.logger.warning("输入数据格式不正确，无法转换为DataFrame")
            return None
        
        try:
            return pd.DataFrame(data)
        except Exception as e:
            self.logger.error(f"转换为DataFrame失败: {e}")
            return None
    
    def validate_json_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        验证JSON数据是否符合模式（简化版）
        Validate if JSON data conforms to schema (simplified version)
        
        Args:
            data: JSON数据 - JSON data
            schema: JSON模式 - JSON schema
            
        Returns:
            bool: 是否符合模式 - Whether it conforms to the schema
        """
        # 这是一个简化的模式验证，实际应用中可使用jsonschema库
        # This is a simplified schema validation, jsonschema library can be used in real applications
        if not isinstance(data, dict) or not isinstance(schema, dict):
            return False
        
        for key, value_type in schema.items():
            if key not in data:
                self.logger.warning(f"JSON数据缺少键: {key}")
                return False
            
            # 简化类型检查 - Simplified type checking
            expected_type = None
            if value_type == "string":
                expected_type = str
            elif value_type == "number":
                expected_type = (int, float)
            elif value_type == "boolean":
                expected_type = bool
            elif value_type == "array":
                expected_type = list
            elif value_type == "object":
                expected_type = dict
            
            if expected_type and not isinstance(data[key], expected_type):
                self.logger.warning(f"JSON数据键 '{key}' 类型错误，期望 {expected_type}, 得到 {type(data[key])}")
                return False
        
        return True
    
    def merge_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并两个字典（dict2覆盖dict1的同名键）
        Merge two dictionaries (dict2 overrides dict1 for same keys)
        
        Args:
            dict1: 第一个字典 - First dictionary
            dict2: 第二个字典 - Second dictionary
            
        Returns:
            Dict[str, Any]: 合并后的字典 - Merged dictionary
        """
        merged = dict1.copy()
        merged.update(dict2)
        return merged
    
    def safe_json_loads(self, json_string: str) -> Optional[Union[Dict, List]]:
        """
        安全地加载JSON字符串
        Safely load JSON string
        
        Args:
            json_string: JSON字符串 - JSON string
            
        Returns:
            Optional[Union[Dict, List]]: 解析后的对象或None - Parsed object or None
        """
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}, 字符串: {json_string[:100]}...")
            return None
        except TypeError:
            self.logger.error(f"JSON解析类型错误，期望字符串，得到: {type(json_string)}")
            return None
            
    def get_nested_value(self, data: Dict[str, Any], path: str, default: Any = None) -> Any:
        """
        安全地获取嵌套字典中的值
        Safely get value from nested dictionary
        
        Args:
            data: 字典数据 - Dictionary data
            path: 路径字符串，例如 "key1.key2.key3" - Path string, e.g., "key1.key2.key3"
            default: 默认值 - Default value
            
        Returns:
            Any: 获取到的值或默认值 - Retrieved value or default value
        """
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list):
                try:
                    idx = int(key)
                    if 0 <= idx < len(current):
                        current = current[idx]
                    else:
                        return default
                except ValueError:
                    return default # 键不是有效的列表索引
            else:
                return default
        
        return current

# 便捷函数 - Convenience functions
_data_utils_instance = None

def get_data_utils() -> DataUtils:
    """获取DataUtils单例 - Get DataUtils singleton instance"""
    global _data_utils_instance
    if _data_utils_instance is None:
        _data_utils_instance = DataUtils()
    return _data_utils_instance

if __name__ == "__main__":
    # 测试数据处理工具 - Test data processing utility
    utils = get_data_utils()
    
    # 清理文本测试 - Clean text test
    text = "  这是  一个   包含 多余 空格 的 文本  "
    cleaned = utils.clean_text(text)
    print(f"清理文本: '{text}' -> '{cleaned}'")
    
    # 日期规范化测试 - Date normalization test
    date_str = "2023年10月5日"
    normalized_date = utils.normalize_date(date_str)
    print(f"日期规范化: '{date_str}' -> '{normalized_date}'")
    
    date_str_fail = "无效日期"
    normalized_date_fail = utils.normalize_date(date_str_fail)
    print(f"日期规范化（失败）: '{date_str_fail}' -> '{normalized_date_fail}'")
    
    # 提取数字测试 - Extract numbers test
    num_text = "价格上涨了10.5%，达到200元，影响了3个行业"
    numbers = utils.extract_numbers(num_text)
    print(f"提取数字: '{num_text}' -> {numbers}")
    
    # 字典转DataFrame测试 - Dict to DataFrame test
    dict_list = [{"name": "A", "value": 1}, {"name": "B", "value": 2}]
    df = utils.dict_to_dataframe(dict_list)
    if df is not None:
        print("DataFrame转换成功:")
        print(df.head())
    
    # JSON模式验证测试 - JSON schema validation test
    data_valid = {"name": "测试", "count": 10}
    schema_valid = {"name": "string", "count": "number"}
    is_valid = utils.validate_json_schema(data_valid, schema_valid)
    print(f"JSON模式验证 (有效): {is_valid}")
    
    data_invalid = {"name": "测试", "count": "十"}
    is_invalid = utils.validate_json_schema(data_invalid, schema_valid)
    print(f"JSON模式验证 (无效): {is_invalid}")
    
    # 获取嵌套值测试 - Get nested value test
    nested_dict = {"a": {"b": {"c": 100, "d": [10, 20]}}}
    value_c = utils.get_nested_value(nested_dict, "a.b.c")
    print(f"获取嵌套值 a.b.c: {value_c}")
    value_d0 = utils.get_nested_value(nested_dict, "a.b.d.0")
    print(f"获取嵌套值 a.b.d.0: {value_d0}")
    value_none = utils.get_nested_value(nested_dict, "a.x.y", default="未找到")
    print(f"获取嵌套值 a.x.y (默认): {value_none}")
    
    print("数据处理工具测试完成")
