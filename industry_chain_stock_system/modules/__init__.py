"""
核心业务模块包初始化文件
Core business modules package initialization
"""

# 版本信息 - Version information
__version__ = "1.0.0"
__author__ = "Industry Chain Stock System Team"

# 模块描述 - Module description
__doc__ = """
核心业务模块包，包含系统的主要业务逻辑和功能模块
Core business modules package containing main business logic and functional modules
"""

# 导入核心模块 - Import core modules
from .message_processor import MessageProcessor
from .industry_chain import IndustryChainBuilder
from .causal_reasoning import CausalReasoningEngine
from .investment_decision import InvestmentAdvisor

# 模块列表 - Module list
__all__ = [
    "MessageProcessor",
    "IndustryChainBuilder", 
    "CausalReasoningEngine",
    "InvestmentAdvisor"
]
