"""
系统配置文件
Configuration file for the industry chain stock system
"""

# API配置 - API Configuration
API_CONFIG = {
    # OpenAI兼容接口配置 - OpenAI Compatible API Configuration
    "base_url": "https://api.deepseek.com/v1",  # API基础URL - API base URL
    "api_key": "sk-28e66466f44148b4b6135f6e92d18651",  # API密钥 - API key
    "model": "deepseek-chat",  # 默认模型 - Default model (恢复为不带路径的名称)
    "temperature": 0.5,  # 温度参数，控制输出的随机性 - Temperature for output randomness
    "timeout": 300,  # 请求超时时间（秒）- Request timeout in seconds
}

# 数据库配置 - Database Configuration
DATABASE_CONFIG = {
    "type": "sqlite",  # 数据库类型 - Database type
    "path": "data/industry_chain.db",  # 数据库文件路径 - Database file path
    "echo": False,  # 是否打印SQL语句 - Whether to print SQL statements
}

# 数据源配置 - Data Source Configuration
DATA_SOURCE_CONFIG = {
    # AKShare配置 - AKShare configuration
    "akshare": {
        "enabled": True,  # 是否启用AKShare - Whether to enable AKShare
        "timeout": 120,  # 请求超时时间 - Request timeout
        "retry_times": 3,  # 重试次数 - Retry times
    },
    
    # 新闻数据源配置 - News data source configuration
    "news_sources": [
        "eastmoney",  # 东方财富 - East Money
        "sina_finance",  # 新浪财经 - Sina Finance
        "163_finance",  # 网易财经 - 163 Finance
    ],
}

# Agent配置 - Agent Configuration
AGENT_CONFIG = {
    # 消息分析Agent配置 - Message Analyst Agent Configuration
    "message_analyst": {
        "role": "资深财经新闻分析师",  # Agent角色 - Agent role
        "goal": "准确解析财经消息，提取关键事件信息",  # Agent目标 - Agent goal
        "backstory": "拥有10年财经新闻分析经验，擅长从复杂信息中提取核心要素",  # Agent背景 - Agent backstory
        "verbose": True,  # 是否输出详细信息 - Whether to output verbose information
    },
    
    # 产业链专家Agent配置 - Industry Chain Expert Agent Configuration
    "industry_expert": {
        "role": "产业链分析专家",  # Agent角色 - Agent role
        "goal": "构建完整的产业链知识图谱，分析上下游关系",  # Agent目标 - Agent goal
        "backstory": "在产业研究领域工作15年，对各行业生态有深入理解",  # Agent背景 - Agent backstory
        "verbose": True,  # 是否输出详细信息 - Whether to output verbose information
    },
    
    # 因果推理专家Agent配置 - Causal Reasoning Expert Agent Configuration
    "causal_expert": {
        "role": "因果关系推理专家",  # Agent角色 - Agent role
        "goal": "构建清晰的因果链条，分析事件影响传导路径",  # Agent目标 - Agent goal
        "backstory": "专注于因果关系研究的资深分析师，善于构建逻辑严谨的推理链条",  # Agent背景 - Agent backstory
        "verbose": True,  # 是否输出详细信息 - Whether to output verbose information
    },
    
    # 投资顾问Agent配置 - Investment Advisor Agent Configuration
    "investment_advisor": {
        "role": "资深投资顾问",  # Agent角色 - Agent role
        "goal": "基于深度分析提供有价值的投资建议",  # Agent目标 - Agent goal
        "backstory": "20年投资经验，擅长将复杂分析转化为可执行的投资策略",  # Agent背景 - Agent backstory
        "verbose": True,  # 是否输出详细信息 - Whether to output verbose information
    },
}

# 系统参数配置 - System Parameters Configuration
SYSTEM_CONFIG = {
    # 推理参数 - Reasoning parameters
    "max_reasoning_depth": 5,  # 最大推理深度 - Maximum reasoning depth
    "confidence_threshold": 0.7,  # 置信度阈值 - Confidence threshold
    "max_stocks_recommendation": 5,  # 最大推荐股票数量 - Maximum number of recommended stocks
    
    # 缓存配置 - Cache configuration
    "enable_cache": True,  # 是否启用缓存 - Whether to enable cache
    "cache_expire_hours": 24,  # 缓存过期时间（小时）- Cache expiration time in hours
    
    # 日志配置 - Logging configuration
    "log_level": "INFO",  # 日志级别 - Log level
    "log_file": "logs/system.log",  # 日志文件路径 - Log file path
}

# 行业分类配置 - Industry Classification Configuration
INDUSTRY_CONFIG = {
    # 申万一级行业分类 - SWICS Level 1 Industry Classification
    "sw_l1_industries": [
        "农林牧渔", "采掘", "化工", "钢铁", "有色金属", "电子", "家用电器",
        "食品饮料", "纺织服装", "轻工制造", "医药生物", "公用事业", "交通运输",
        "房地产", "商业贸易", "休闲服务", "综合", "建筑材料", "建筑装饰",
        "电气设备", "国防军工", "计算机", "传媒", "通信", "银行", "非银金融",
        "汽车", "机械设备"
    ],
    
    # 重点关注的新兴行业 - Key emerging industries to focus on
    "emerging_industries": [
        "新能源", "新能源汽车", "人工智能", "大数据", "云计算", "物联网",
        "5G通信", "半导体", "生物医药", "新材料", "节能环保", "数字经济"
    ],
}

# 风险控制配置 - Risk Control Configuration
RISK_CONFIG = {
    # 风险等级定义 - Risk level definition
    "risk_levels": ["低", "中", "高"],  # Risk levels: Low, Medium, High
    
    # 风险因子权重 - Risk factor weights
    "risk_factors": {
        "market_volatility": 0.3,  # 市场波动性 - Market volatility
        "industry_concentration": 0.2,  # 行业集中度 - Industry concentration
        "policy_sensitivity": 0.2,  # 政策敏感性 - Policy sensitivity
        "liquidity_risk": 0.15,  # 流动性风险 - Liquidity risk
        "fundamental_risk": 0.15,  # 基本面风险 - Fundamental risk
    },
}

# 输出格式配置 - Output Format Configuration
OUTPUT_CONFIG = {
    # 支持的输出格式 - Supported output formats
    "supported_formats": ["json", "text", "html", "pdf"],
    
    # 默认输出格式 - Default output format
    "default_format": "json",
    
    # 报告模板配置 - Report template configuration
    "report_templates": {
        "analysis_report": "templates/analysis_report.html",
        "investment_advice": "templates/investment_advice.html",
        "causal_chain": "templates/causal_chain.html",
    },
}
