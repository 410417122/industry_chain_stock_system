"""
Agent模块包初始化文件
Agent modules package initialization
"""

# 版本信息 - Version information
__version__ = "1.0.0"
__author__ = "Industry Chain Stock System Team"

# 模块描述 - Module description
__doc__ = """
Agent模块包，包含系统中使用的AI Agent定义和管理功能
Agent modules package containing AI Agent definitions and management functionality used in the system
"""

# 导入Agent管理器 - Import Agent manager
from .crew_manager import CrewManager

# 模块列表 - Module list
__all__ = [
    "CrewManager"
]
