"""
基础工具模块
Basic Tools Module

包含从核心模块功能转换而来的基础Agent工具
Contains basic Agent tools converted from core module functionalities
"""

import json # 导入json库，用于处理JSON数据
from crewai.tools import BaseTool # 从crewai.tools导入BaseTool，用于创建自定义工具
from pydantic import ConfigDict # 导入ConfigDict用于模型配置
from pathlib import Path # 导入Path，用于处理文件路径
import sys # 导入sys，用于系统相关操作

# 将项目根目录添加到Python路径，以便正确导入模块
# Add project root directory to Python path for correct module imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# 从项目中导入消息处理器模块
# Import the MessageProcessor module from the project
from modules.message_processor import MessageProcessor # 从消息处理模块导入MessageProcessor类
from modules.industry_chain import IndustryChainBuilder # 从产业链模块导入IndustryChainBuilder类
from modules.causal_reasoning import CausalReasoningEngine # 从因果推理模块导入CausalReasoningEngine类
from modules.investment_decision import InvestmentAdvisor # 从投资决策模块导入InvestmentAdvisor类

class MessageAnalysisTool(BaseTool):
    """
    消息分析工具
    Message Analysis Tool

    用于解析财经消息并提取关键事件信息。
    Used to parse financial messages and extract key event information.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True) # 允许任意类型作为字段

    name: str = "消息分析工具" # 工具的名称
    description: str = "解析财经消息并提取关键事件信息。输入财经消息文本，输出结构化的事件信息JSON字符串。" # 工具的描述
    message_processor: MessageProcessor # 将message_processor声明为字段

    # Pydantic会自动处理在实例化时传递的同名字段的赋值
    # __init__ 可以保持原样，或者如果为了清晰，可以移除 message_processor 参数，
    # 因为Pydantic会通过kwargs处理它。但保留它并显式赋值也是可以的。
    # 为了最小化更改并保持显式性，我们保留__init__。

    def __init__(self, message_processor: MessageProcessor, **kwargs):
        """
        初始化消息分析工具。
        Initialize the Message Analysis Tool.

        Args:
            message_processor (MessageProcessor): 消息处理器实例。
                                                  An instance of the MessageProcessor.
        """
        # 将 message_processor 传递给 Pydantic 的 BaseModel 初始化
        # Pass message_processor to Pydantic's BaseModel initialization
        super().__init__(message_processor=message_processor, **kwargs)
        # self.message_processor = message_processor # Pydantic 会自动处理这个赋值

    def _run(self, message_text: str) -> str:
        """
        执行消息分析。
        Execute message analysis.

        Args:
            message_text (str): 要分析的财经消息文本。
                                The financial message text to analyze.

        Returns:
            str: 结构化的事件信息JSON字符串。
                 A JSON string of the structured event information.
        """
        # 调用消息处理器的parse_message方法进行消息解析
        # Call the parse_message method of the message processor to parse the message
        event_info = self.message_processor.parse_message(message_text)
        # 将解析结果转换为JSON字符串并返回，确保中文字符正常显示
        # Convert the parsing result to a JSON string and return, ensuring Chinese characters are displayed correctly
        return json.dumps(event_info, ensure_ascii=False)

# 可以在此文件中继续添加其他基础工具类，例如 IndustryChainTool, CausalReasoningTool, InvestmentAdvisorTool
# You can continue to add other basic tool classes in this file, such as IndustryChainTool, CausalReasoningTool, InvestmentAdvisorTool

class IndustryChainTool(BaseTool):
    """
    产业链构建工具
    Industry Chain Tool

    用于根据行业代码构建完整的产业链知识图谱。
    Used to build a complete industry chain knowledge graph based on industry code.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True) # 允许任意类型作为字段

    name: str = "产业链构建工具" # 工具的名称
    description: str = "根据行业代码构建完整的产业链知识图谱。输入行业代码，输出产业链数据JSON字符串。" # 工具的描述
    industry_chain_builder: IndustryChainBuilder # 将industry_chain_builder声明为字段

    def __init__(self, industry_chain_builder: IndustryChainBuilder, **kwargs):
        """
        初始化产业链构建工具。
        Initialize the Industry Chain Tool.

        Args:
            industry_chain_builder (IndustryChainBuilder): 产业链构建器实例。
                                                            An instance of the IndustryChainBuilder.
        """
        super().__init__(industry_chain_builder=industry_chain_builder, **kwargs) # 将其传递给Pydantic

    def _run(self, industry_code: str) -> str:
        """
        执行产业链构建。
        Execute industry chain building.

        Args:
            industry_code (str): 要构建产业链的行业代码。
                                 The industry code for which to build the industry chain.

        Returns:
            str: 产业链数据JSON字符串。
                 A JSON string of the industry chain data.
        """
        # 调用产业链构建器的build_industry_chain方法进行产业链构建
        # Call the build_industry_chain method of the industry chain builder to build the industry chain
        industry_chain_data = self.industry_chain_builder.build_industry_chain(industry_code)
        # 将构建结果转换为JSON字符串并返回，确保中文字符正常显示
        # Convert the building result to a JSON string and return, ensuring Chinese characters are displayed correctly
        return json.dumps(industry_chain_data, ensure_ascii=False)

class CausalReasoningTool(BaseTool):
    """
    因果推理图构建工具
    Causal Reasoning Graph Building Tool

    用于结合事件信息和产业链数据，构建因果推理图。
    Used to build a causal reasoning graph by combining event information and industry chain data.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True) # 允许任意类型作为字段

    name: str = "因果推理图构建工具" # 工具的名称
    description: str = "结合事件信息JSON字符串和产业链数据JSON字符串，构建因果推理图，输出因果图JSON字符串。" # 工具的描述
    causal_reasoning_engine: CausalReasoningEngine # 将causal_reasoning_engine声明为字段

    def __init__(self, causal_reasoning_engine: CausalReasoningEngine, **kwargs):
        """
        初始化因果推理图构建工具。
        Initialize the Causal Reasoning Graph Building Tool.

        Args:
            causal_reasoning_engine (CausalReasoningEngine): 因果推理引擎实例。
                                                              An instance of the CausalReasoningEngine.
        """
        super().__init__(causal_reasoning_engine=causal_reasoning_engine, **kwargs) # 将其传递给Pydantic

    def _run(self, event_info_json: str, industry_chain_json: str) -> str:
        """
        执行因果推理图构建。
        Execute causal reasoning graph building.

        Args:
            event_info_json (str): 事件信息的JSON字符串。
                                   A JSON string of the event information.
            industry_chain_json (str): 产业链数据的JSON字符串。
                                       A JSON string of the industry chain data.

        Returns:
            str: 因果图JSON字符串。
                 A JSON string of the causal graph.
        """
        # 将输入的JSON字符串转换为Python字典
        # Convert the input JSON strings to Python dictionaries
        event_info = json.loads(event_info_json)
        industry_chain = json.loads(industry_chain_json)
        # 调用因果推理引擎的create_causal_graph方法构建因果图
        # Call the create_causal_graph method of the causal reasoning engine to build the causal graph
        causal_graph_data = self.causal_reasoning_engine.create_causal_graph(event_info, industry_chain)
        # 将构建结果转换为JSON字符串并返回，确保中文字符正常显示
        # Convert the building result to a JSON string and return, ensuring Chinese characters are displayed correctly
        return json.dumps(causal_graph_data, ensure_ascii=False)

class InvestmentAdvisorTool(BaseTool):
    """
    投资建议生成工具
    Investment Advisor Tool

    用于基于因果分析和产业链数据，生成投资建议。
    Used to generate investment advice based on causal analysis and industry chain data.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True) # 允许任意类型作为字段

    name: str = "投资建议生成工具" # 工具的名称
    description: str = "基于因果图JSON字符串和产业链数据JSON字符串，生成投资建议JSON字符串。" # 工具的描述
    investment_advisor: InvestmentAdvisor # 将investment_advisor声明为字段

    def __init__(self, investment_advisor: InvestmentAdvisor, **kwargs):
        """
        初始化投资建议生成工具。
        Initialize the Investment Advisor Tool.

        Args:
            investment_advisor (InvestmentAdvisor): 投资顾问实例。
                                                    An instance of the InvestmentAdvisor.
        """
        super().__init__(investment_advisor=investment_advisor, **kwargs) # 将其传递给Pydantic

    def _run(self, causal_graph_json: str, industry_chain_json: str) -> str:
        """
        执行投资建议生成。
        Execute investment advice generation.

        Args:
            causal_graph_json (str): 因果图的JSON字符串。
                                     A JSON string of the causal graph.
            industry_chain_json (str): 产业链数据的JSON字符串。
                                       A JSON string of the industry chain data.

        Returns:
            str: 投资建议JSON字符串。
                 A JSON string of the investment recommendations.
        """
        # 将输入的JSON字符串转换为Python字典
        # Convert the input JSON strings to Python dictionaries
        causal_graph = json.loads(causal_graph_json)
        industry_chain = json.loads(industry_chain_json)
        # 调用投资顾问的generate_recommendations方法生成投资建议
        # Call the generate_recommendations method of the investment advisor to generate investment recommendations
        recommendations = self.investment_advisor.generate_recommendations(causal_graph, industry_chain)
        # 将建议结果转换为JSON字符串并返回，确保中文字符正常显示
        # Convert the recommendation results to a JSON string and return, ensuring Chinese characters are displayed correctly
        return json.dumps(recommendations, ensure_ascii=False)
