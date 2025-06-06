"""
Agent协作管理器模块
Agent Collaboration Manager Module

使用CrewAI框架管理和协调多个AI Agent的工作流程，集成增强型工具提升系统能力
Manages and coordinates the workflow of multiple AI Agents using the CrewAI framework,
integrating enhanced tools to improve system capabilities
"""

import json # 导入json库，用于处理JSON数据
from typing import Dict, List, Any # 导入类型提示相关的模块
from pathlib import Path # 导入Path，用于处理文件路径
from datetime import datetime # 导入datetime模块中的datetime类

# 导入CrewAI框架 - Import CrewAI framework
from crewai import Agent, Task, Crew, Process, LLM # 导入LLM相关的类
from crewai.crews.crew_output import CrewOutput # 导入CrewOutput，用于处理Crew的输出
from crewai.tasks.task_output import TaskOutput # 导入TaskOutput，用于处理Task的输出
from crewai_tools import SerperDevTool # 直接从crewai_tools导入SerperDevTool
from langchain_community.utilities import GoogleSerperAPIWrapper # 导入GoogleSerperAPIWrapper以配置超时

# 导入工具模块 - Import utility modules
import sys # 导入sys，用于系统相关操作
# 将项目根目录添加到Python路径，以便正确导入模块
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logger import LoggerMixin, log_execution_time # 从日志工具导入LoggerMixin和log_execution_time装饰器
from utils.api_client import LLMClient # 从API客户端工具导入LLMClient

# 导入核心模块 - Import core modules
from modules.message_processor import MessageProcessor # 从消息处理模块导入MessageProcessor类
from modules.industry_chain import IndustryChainBuilder # 从产业链模块导入IndustryChainBuilder类
from modules.causal_reasoning import CausalReasoningEngine # 从因果推理模块导入CausalReasoningEngine类
from modules.investment_decision import InvestmentAdvisor # 从投资决策模块导入InvestmentAdvisor类

# 导入所有工具类
# Import all tool classes
from tools.basic_tools import MessageAnalysisTool, IndustryChainTool, CausalReasoningTool, InvestmentAdvisorTool # 导入基础工具
from tools.causal_tools import EnhancedCausalGraphTool # 导入增强型因果图工具
from tools.enhanced_tools import ( # 导入所有增强型工具
    IndustrySearchTool, StockDataTool, # SerperSearchTool 已被移除，将使用 crewai_tools.SerperDevTool
    IndustryStockTool, StockScreeningTool,
    RelevanceAnalysisTool, MacroDataTool
)

# 导入配置文件 - Import configuration
from config import AGENT_CONFIG, API_CONFIG # 从配置文件导入Agent和API的配置

import os # 导入os模块以设置环境变量

# Serper API 密钥常量
# 注意：更安全的做法是将密钥存储在.env文件或专门的配置管理系统中，而不是硬编码。
# 但根据当前上下文和用户提供的密钥，我们暂时在此定义。
_SERPER_API_KEY_CONST = "4ceedf042924f2d3c5e0b750bff4abf165109a10"

class CrewManager(LoggerMixin): # 定义CrewManager类，继承LoggerMixin以使用日志功能
    """
    Agent协作管理器类
    Agent Collaboration Manager Class
    
    负责创建、配置和运行Agent团队
    Responsible for creating, configuring, and running Agent crews
    """
    
    def __init__(self, llm_client: LLMClient): # CrewManager的构造函数
        """
        初始化Agent协作管理器
        Initialize Agent Collaboration Manager
        
        Args:
            llm_client: LLM客户端实例 - LLM client instance
        """
        super().__init__() # 调用父类的构造函数，初始化日志记录器
        self.llm_client = llm_client # 保存LLM客户端实例
        
        # 初始化核心模块 - Initialize core modules
        self.message_processor = MessageProcessor(self.llm_client) # 初始化消息处理器
        self.industry_chain_builder = IndustryChainBuilder(self.llm_client) # 初始化产业链构建器
        self.causal_reasoning_engine = CausalReasoningEngine(self.llm_client) # 初始化因果推理引擎
        self.investment_advisor = InvestmentAdvisor(self.llm_client) # 初始化投资顾问

        # 初始化基础工具实例 - Initialize basic tool instances
        self.message_analysis_tool = MessageAnalysisTool(message_processor=self.message_processor) # 初始化消息分析工具
        self.industry_chain_tool = IndustryChainTool(industry_chain_builder=self.industry_chain_builder) # 初始化产业链构建工具
        self.enhanced_causal_graph_tool = EnhancedCausalGraphTool(causal_reasoning_engine=self.causal_reasoning_engine) # 初始化增强型因果图工具
        self.investment_advisor_tool = InvestmentAdvisorTool(investment_advisor=self.investment_advisor) # 初始化投资建议工具
        
        # 初始化增强型工具实例 - Initialize enhanced tool instances
        # 在实例化SerperDevTool之前，确保SERPER_API_KEY环境变量已设置
        if not os.getenv("SERPER_API_KEY") and _SERPER_API_KEY_CONST:
            os.environ["SERPER_API_KEY"] = _SERPER_API_KEY_CONST # 确保环境变量被设置，GoogleSerperAPIWrapper默认会读取它
            self.logger.info("SERPER_API_KEY 环境变量已通过模块内常量设置。")
        
        serper_api_key_to_use = os.getenv("SERPER_API_KEY") or _SERPER_API_KEY_CONST

        if not serper_api_key_to_use:
            self.logger.warning("SERPER_API_KEY 未设置（环境变量和内部常量均无）。SerperDevTool 将使用一个虚拟密钥，可能无法正常工作。")
            # 使用一个虚拟的API密钥，以便api_wrapper可以被实例化，但实际调用会失败
            configured_serper_wrapper = GoogleSerperAPIWrapper(serper_api_key="DUMMY_INVALID_KEY", timeout=120)
        else:
            self.logger.info(f"为 SerperDevTool 配置 API Key 和 Timeout (120s)。")
            configured_serper_wrapper = GoogleSerperAPIWrapper(
                serper_api_key=serper_api_key_to_use, # 可以直接传递key
                timeout=120 # 设置超时时间为120秒
            )
        
        self.serper_dev_tool = SerperDevTool(api_wrapper=configured_serper_wrapper) # 使用配置好的wrapper实例化SerperDevTool
        
        self.industry_search_tool = IndustrySearchTool(search_tool=self.serper_dev_tool) # 初始化行业搜索工具，依赖标准的SerperDevTool
        self.stock_data_tool = StockDataTool() # 初始化股票数据工具
        self.industry_stock_tool = IndustryStockTool() # 初始化行业股票工具
        self.stock_screening_tool = StockScreeningTool() # 初始化股票筛选工具
        self.relevance_analysis_tool = RelevanceAnalysisTool(search_tool=self.serper_dev_tool) # 初始化相关性分析工具，依赖标准的SerperDevTool
        self.macro_data_tool = MacroDataTool() # 初始化宏观数据工具
        
        # 创建Agent实例 - Create Agent instances
        self.agents = self._create_agents() # 调用内部方法创建所有Agent
        
        self.logger.info("Agent协作管理器初始化完成") # 记录日志：Agent协作管理器初始化完成

    def _create_agents(self) -> Dict[str, Agent]: # 定义创建Agent的内部方法
        """
        创建并配置Agent实例
        Create and configure Agent instances
        
        Returns:
            Dict[str, Agent]: Agent实例字典 - Dictionary of Agent instances
        """
        agents = {} # 初始化Agent字典
        
        # 为Agent创建一个共享的LLM实例
        # 使用CrewAI的LLM类进行配置
        shared_llm = LLM( # 创建共享的LLM实例
            model="deepseek/deepseek-chat",  # CrewAI中明确使用 "deepseek/deepseek-chat"
            api_key=API_CONFIG.get("api_key"), # API密钥从配置中读取
            config={ # LLM的特定于模型的配置
                "base_url": API_CONFIG.get("base_url"), # API基础URL
                "temperature": API_CONFIG.get("temperature", 0.5), # 温度参数
                "timeout": API_CONFIG.get("timeout", 120) # 更新默认回退值为120秒
            }
        )

        # 消息分析Agent - Message Analyst Agent
        msg_analyst_config = AGENT_CONFIG.get("message_analyst", {}) # 从配置中获取消息分析Agent的配置
        agents["message_analyst"] = Agent( # 创建消息分析Agent实例
            role="""
你是一位资深的财经新闻传导分析专家，拥有15年A股市场研究经验。
你的专长是从复杂的财经新闻中精准识别出对A股行业板块的具体影响路径。
你深度了解申万行业分类体系，能够准确判断新闻事件对不同行业板块的传导机制。
""", # Agent角色
            goal="""
从给定的财经新闻中，精准分析出：
1. 核心事件的影响因子和传导机制
2. 受影响的A股行业板块（精确到申万二级分类）
3. 影响的传导路径、强度、方向和时间窗口
4. 为后续的产业链分析提供精准的行业标的
""", # Agent目标
            backstory=( # Agent背景故事，进行详细描述
                "拥有超过10年的财经新闻分析经验，对市场动态有敏锐洞察力。"
                "擅长从海量信息中快速识别核心事件，并结合多方信息源进行综合研判，"
                "为后续的产业链分析和投资决策提供坚实的信息基础。"
            ),
            tools=[self.message_analysis_tool, self.serper_dev_tool], # Agent直接使用标准的SerperDevTool
            llm=shared_llm, # 为Agent配置LLM实例
            verbose=msg_analyst_config.get("verbose", True), # 是否输出详细日志
            allow_delegation=False, # 是否允许任务委派
            max_iter=5 # 设置最大迭代次数
        )
        
        # 产业链专家Agent - Industry Chain Expert Agent
        industry_expert_config = AGENT_CONFIG.get("industry_expert", {}) # 从配置中获取产业链专家Agent的配置
        agents["industry_expert"] = Agent( # 创建产业链专家Agent实例
            role="""
你是一位产业链深度研究专家，在中国A股市场产业链分析领域有20年经验。
你精通各行业的完整产业链结构，深度了解上中下游的供应关系、技术依赖和价值传导。
你擅长将宏观的行业影响精准分解到具体的产业链环节和代表性企业。
""", # Agent角色
            goal="""
基于前序的行业影响分析，在确定的重点行业内构建详细的产业链图谱：
1. 绘制完整的上中下游产业链结构
2. 识别每个环节的关键企业和A股标的
3. 分析事件对各产业链环节的差异化影响
4. 构建清晰的"事件→行业→产业链环节→具体企业"传导路径
""", # Agent目标
            backstory=( # Agent背景故事，进行详细描述
                "在产业经济研究领域拥有15年经验，主导过多个国家级产业规划项目。"
                "对中国各主要行业的生态系统有深刻理解，能够快速构建和验证产业链模型，"
                "并利用专业数据工具进行深度分析，确保产业链分析的准确性和前瞻性。"
            ),
            tools=[ # Agent使用的工具列表
                self.industry_chain_tool, 
                self.industry_search_tool, 
                self.industry_stock_tool,  
                self.stock_data_tool       
            ],
            llm=shared_llm, # 为Agent配置LLM实例
            verbose=industry_expert_config.get("verbose", True), # 是否输出详细日志
            allow_delegation=False, # 是否允许任务委派
            max_iter=7 # 设置最大迭代次数
        )
        
        # 因果推理专家Agent - Causal Reasoning Expert Agent
        causal_expert_config = AGENT_CONFIG.get("causal_expert", {}) # 从配置中获取因果推理专家Agent的配置
        agents["causal_expert"] = Agent( # 创建因果推理专家Agent实例
            role=causal_expert_config.get("role", "首席因果逻辑分析师"), # Agent角色
            goal=( # Agent目标，进行详细描述
                "构建清晰、可解释、多层次的因果传导链条。精确分析特定事件对产业链各环节的直接影响和间接传导效应，"
                "量化影响程度，识别关键风险点和机遇点，并提供动态的可视化结果和核心传导路径解读。"
            ),
            backstory=( # Agent背景故事，进行详细描述
                "专注于复杂系统因果关系建模与分析的资深专家，拥有经济学和数据科学双博士学位。"
                "逻辑思维严谨，擅长将抽象的因果关系转化为直观的可视化模型，并结合宏观经济数据和市场动态进行综合判断，"
                "确保因果分析的深度、准确性和实用性。"
            ),
            tools=[ # Agent使用的工具列表
                self.enhanced_causal_graph_tool, 
                self.macro_data_tool 
            ],
            llm=shared_llm, # 为Agent配置LLM实例
            verbose=causal_expert_config.get("verbose", True), # 是否输出详细日志
            allow_delegation=False, # 是否允许任务委派
            max_iter=7 # 设置最大迭代次数
        )

        # 股票池构建专家Agent - Stock Pool Builder Agent
        stock_pool_builder_config = AGENT_CONFIG.get("stock_pool_builder", {}) # 新增Agent配置获取
        agents["stock_pool_builder"] = Agent(
            role="""
你是一位专业的A股投资标的筛选专家，精通基于产业链分析构建投资标的池。
你擅长将复杂的产业链分析转化为结构化的、可投资的股票池，并进行科学的分层分级。
你深度了解A股市场特点，能够准确评估不同股票与投资主题的相关性。
""",
            goal="""
基于前序的产业链分析，构建结构化的股票池：
1. 收集产业链覆盖的所有相关A股标的
2. 按相关性和投资价值进行科学分层
3. 为每只股票提供详细的投资逻辑分析
4. 构建"高-中-低"相关性的分层股票池
""",
            backstory="""
在多家头部券商研究所担任高级策略分析师超过10年，专注于A股市场行业研究和个股挖掘。
主导构建了多个行业主题股票池，以逻辑严谨、覆盖全面、实用性强著称。
对产业链上下游联动和价值传导有深刻理解，能够精准识别投资机会。
""",
            tools=[
                self.stock_data_tool,
                self.stock_screening_tool,
                self.industry_stock_tool # 可能需要此工具来获取行业下的所有股票
            ],
            llm=shared_llm,
            verbose=stock_pool_builder_config.get("verbose", True),
            allow_delegation=False,
            max_iter=8
        )
        
        # 投资顾问Agent - Investment Advisor Agent
        investment_advisor_config = AGENT_CONFIG.get("investment_advisor", {}) # 从配置中获取投资顾问Agent的配置
        agents["investment_advisor"] = Agent( # 创建投资顾问Agent实例
            role="""
你是一位基于传导分析的投资决策专家，拥有超过20年的A股市场实战投资经验和丰富的量化策略开发背景。
你擅长将复杂的市场分析和事件驱动因素转化为可执行的投资策略，注重风险管理和组合优化。
你的核心能力是从结构化的股票池中，结合完整的"新闻→行业→产业链→股票"逻辑链，筛选出最优投资标的。
""", # Agent角色
            goal="""
基于前序构建的分层股票池和完整的传导分析逻辑，进行最终的投资决策：
1. 从高、中相关性股票池中筛选出最具投资价值的核心标的。
2. 为每个推荐标的提供清晰、可追溯的完整投资逻辑链。
3. 详细阐述每个标的的潜在催化剂和主要风险点。
4. 构建一个风险可控、具有吸引力的最终投资组合建议。
""", # Agent目标
            backstory=( # Agent背景故事，进行详细描述
                "拥有超过20年的A股市场实战投资经验和丰富的量化策略开发背景。"
                "擅长将复杂的市场分析和事件驱动因素转化为可执行的投资策略，注重风险管理和组合优化，"
                "致力于为投资者提供专业、理性、具有前瞻性的投资决策支持。"
            ),
            tools=[ # Agent使用的工具列表
                self.investment_advisor_tool,
                self.stock_screening_tool,    
                self.relevance_analysis_tool, 
                self.stock_data_tool          
            ],
            llm=shared_llm, # 为Agent配置LLM实例
            verbose=investment_advisor_config.get("verbose", True), # 是否输出详细日志
            allow_delegation=False, # 是否允许任务委派
            max_iter=10 # 设置最大迭代次数
        )
        
        self.logger.info(f"成功创建 {len(agents)} 个Agent实例") # 记录日志：成功创建Agent实例的数量
        return agents # 返回创建的Agent字典
    
    @log_execution_time # 使用装饰器记录方法执行时间
    def run_analysis_crew(self, message_text: str) -> Dict[str, Any]: # 定义运行分析Crew的方法
        """
        运行Agent团队进行分析
        Run Agent crew for analysis
        
        Args:
            message_text: 财经消息文本 - Financial message text
            
        Returns:
            Dict[str, Any]: 分析结果 - Analysis result
        """
        self.logger.info(f"启动Agent团队分析流程，输入消息: {message_text[:100]}...") # 记录日志：启动分析流程和部分输入消息
        
        # 定义任务 - Define tasks
        tasks = self._create_tasks(message_text) # 调用内部方法创建任务
        
        # 创建Crew - Create Crew
        analysis_crew = Crew( # 创建Crew实例
            agents=list(self.agents.values()), # Crew中的Agent列表
            tasks=list(tasks.values()), # Crew中的Task列表
            process=Process.sequential,  # 任务执行方式：顺序执行 - Sequential task execution
            verbose=True, # 将verbose参数修改为布尔值True，以启用详细输出
            # memory=True # 可以考虑为Crew启用记忆功能，以便在任务间共享更复杂的上下文
        )
        
        # 启动Crew执行 - Kick off Crew execution
        try:
            # CrewAI的kickoff方法接受字典输入
            # 我们将消息文本作为输入传递给第一个任务
            # CrewAI会自动处理任务间的上下文传递
            
            # 构造初始输入，确保与第一个任务的期望输入匹配
            # 这里的 'initial_message' 键名需要与Task定义中可能引用的上下文变量名一致
            # (虽然当前Task定义中未使用占位符，但这是良好实践)
            
            result = analysis_crew.kickoff(inputs={"initial_message": message_text}) # 启动Crew执行
            
            self.logger.info("Agent团队分析完成") # 记录日志：Agent团队分析完成
            
            # CrewAI的最终结果通常是最后一个任务的输出
            # 我们需要将其包装成系统期望的格式
            
            # 尝试从CrewAI的输出中提取结构化数据
            # CrewOutput 对象包含 raw, json_dict, pydantic_object 等属性
            if isinstance(result, CrewOutput):
                final_output = result.json_dict # 尝试获取JSON字典格式的输出
                if not final_output and result.raw: # 如果json_dict为空，尝试解析原始输出
                    try:
                        final_output = json.loads(result.raw)
                    except json.JSONDecodeError:
                        self.logger.warning("无法将Crew原始输出解析为JSON，将使用原始字符串。")
                        final_output = {"raw_result": result.raw} # 如果解析失败，则返回原始字符串
            elif isinstance(result, str): # 如果结果直接是字符串
                try:
                    final_output = json.loads(result)
                except json.JSONDecodeError:
                    self.logger.warning("Crew返回的字符串不是有效的JSON，将使用原始字符串。")
                    final_output = {"raw_result": result}
            elif isinstance(result, dict): # 如果结果已经是字典
                final_output = result
            else: # 其他未知类型
                self.logger.warning(f"Agent团队返回结果格式未知: {type(result)}，将尝试转换为字符串。")
                final_output = {"raw_result": str(result)}

            # 构建最终结果
            final_result_package = {
                "success": True,
                "analysis_mode": "agent_crew",
                "timestamp": datetime.now().isoformat(),
                "final_investment_advice": final_output, # 将处理后的输出放入
                "crew_execution_stats": analysis_crew.usage_metrics, # 获取CrewAI的用量统计
                "summary": self._generate_agent_summary(final_output) # 基于处理后的输出生成摘要
            }
            
            return final_result_package # 返回最终结果包

        except Exception as e: # 捕获执行过程中的异常
            self.logger.error(f"Agent团队分析失败: {e}", exc_info=True) # 记录错误日志，包括异常信息
            return { # 返回失败结果
                "success": False,
                "error": "Agent团队分析执行过程中发生严重错误",
                "error_details": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def _create_tasks(self, message_text: str) -> Dict[str, Task]: # 定义创建任务的内部方法
        """
        创建Agent任务
        Create Agent tasks
        
        Args:
            message_text: 初始财经消息文本 - Initial financial message text
            
        Returns:
            Dict[str, Task]: 任务实例字典 - Dictionary of Task instances
        """
        tasks = {} # 初始化任务字典
        
        # 任务1：深度消息解析与事件定义
        tasks["message_analysis"] = Task( # 创建消息分析任务
            description=f"""
请深入分析以下财经消息：'{message_text}'

**第一步：核心事件解构**
请提取：
- 事件核心：用一句话概括核心事件
- 事件类型：政策变化/技术突破/市场供需/突发事件/监管变化/其他
- 关键影响因子：列出3-5个最关键的影响因子（如"补贴减少"、"技术突破"、"需求增长"等）
- 影响性质：正面/负面/中性/混合

**第二步：A股行业板块影响分析**
基于事件分析，请确定受影响的A股行业板块：

**主要影响行业（直接受益/受损）：**
- 使用申万二级行业分类（如801080汽车整车、801230电池等）
- 每个行业必须包含：
  * 申万代码和行业名称
  * 影响评分（0-1分，保留2位小数）
  * 影响方向（positive/negative/mixed）
  * 具体传导机制（详细说明"为什么"这个行业会受到影响）
  * 影响预期强度（高/中/低）

**次要影响行业（间接受益/受损）：**
- 同样格式，但影响评分通常较低
- 重点说明间接传导的逻辑

**第三步：传导时间分析**
请分析影响的时间窗口：
- 即时影响（1-7天）：市场情绪和预期变化
- 短期影响（1-3个月）：政策落地、订单变化、业绩预期调整
- 中期影响（3-12个月）：实际业绩体现、行业格局变化
- 长期影响（1年以上）：产业结构性变化

**第四步：风险因素识别**
识别可能影响传导效果的风险因素：
- 政策执行风险
- 市场反应超预期/不及预期风险
- 竞争格局变化风险
- 宏观环境变化风险
""", # 任务描述
            agent=self.agents["message_analyst"], # 指定执行此任务的Agent
            expected_output="""
一个完整的JSON字符串，包含事件分析、行业影响映射、时间分析、风险因素和分析摘要。
重点确保行业影响映射部分的精准性和传导机制的清晰性。
严格按照以下JSON格式输出：
{{
  "event_analysis": {{
    "core_event": "事件核心一句话描述",
    "event_type": "事件类型",
    "key_factors": ["因子1", "因子2", "因子3"],
    "impact_nature": "正面/负面/中性/混合"
  }},
  "industry_impact_mapping": {{
    "primary_industries": [
      {{
        "sw_code": "801XXX",
        "industry_name": "行业名称",
        "impact_score": 0.XX,
        "impact_direction": "positive/negative/mixed",
        "transmission_mechanism": "详细的传导机制说明，至少50字",
        "impact_intensity": "高/中/低",
        "confidence_level": "高/中/低"
      }}
    ],
    "secondary_industries": [
      {{
        "sw_code": "801XXX", 
        "industry_name": "行业名称",
        "impact_score": 0.XX,
        "impact_direction": "positive/negative/mixed",
        "transmission_mechanism": "间接传导机制说明",
        "impact_intensity": "高/中/低",
        "confidence_level": "高/中/低"
      }}
    ]
  }},
  "temporal_analysis": {{
    "immediate_impact": "即时影响描述",
    "short_term_impact": "短期影响描述", 
    "medium_term_impact": "中期影响描述",
    "long_term_impact": "长期影响描述",
    "peak_impact_timing": "预计影响峰值时间"
  }},
  "risk_factors": [
    {{
      "risk_type": "风险类型",
      "risk_description": "风险描述",
      "impact_on_analysis": "对分析结果的影响"
    }}
  ],
  "analysis_summary": {{
    "total_affected_industries": "受影响行业总数",
    "highest_impact_industry": "影响最大的行业",
    "key_transmission_logic": "核心传导逻辑一句话总结"
  }},
  "visualization_data": {{
    "flowchart_data_for_news_to_industry": {{
        "steps": [
            {{"id": "news_event", "label": "新闻事件核心", "type": "io"}},
            {{"id": "factor1", "label": "影响因子1", "type": "process"}},
            {{"id": "industry1", "label": "主要影响行业1", "type": "output"}}
        ],
        "connections": [
            {{"from": "news_event", "to": "factor1"}},
            {{"from": "factor1", "to": "industry1"}}
        ]
    }}
  }}
}}
"""
        )
        
        # 任务2：产业链深度分析与构建
        tasks["industry_chain_building"] = Task( # 创建产业链构建任务
            description="""
基于上一步的行业影响分析结果，请对影响评分最高的前2个行业进行深度产业链分析。

**分析目标行业：** [从上一步结果中自动提取评分最高的行业]

**第一步：产业链结构映射**
为目标行业构建完整的产业链图谱：

**上游环节分析：**
- 列出3-5个主要上游环节
- 每个环节包含：
  * 环节名称（如"锂电池正极材料"、"汽车芯片"等）
  * 环节描述（该环节的主要功能和价值）
  * 技术壁垒（高/中/低）
  * 市场集中度（高/中/低）
  * 主要参与者类型（原材料供应商、设备商、技术提供商等）

**中游环节分析：**
- 列出2-4个主要中游环节
- 同样的分析维度
- 特别关注制造、组装、集成等核心环节

**下游环节分析：**
- 列出2-4个主要下游环节  
- 关注应用、销售、服务等环节

**第二步：关键企业映射**
为每个产业链环节找出对应的A股上市公司：

**企业筛选标准：**
- 必须是A股上市公司
- 在该环节有重要地位（龙头、重要参与者、新兴力量）
- 业务与该环节高度相关（主营业务占比>30%或战略布局明确）

**企业信息要求：**
- 股票代码和名称
- 在产业链中的精确定位
- 市场地位（龙头/重要参与者/新兴企业）
- 核心业务描述
- 竞争优势简述

**第三步：事件影响传导分析**
分析原始事件对各产业链环节的具体影响：

**传导路径构建：**
- 绘制"事件→一级传导→二级传导→三级传导"的完整路径
- 每个传导步骤要有清晰的逻辑和机制说明
- 标注传导强度和时间延迟

**差异化影响分析：**
- 分析为什么不同环节受到的影响不同
- 识别受益最大和受损最大的环节
- 预测影响的持续时间和强度变化

**第四步：关键节点识别**
识别产业链中的关键控制点：
- 技术瓶颈点
- 供应瓶颈点  
- 需求放大点
- 政策敏感点
""", # 任务描述
            agent=self.agents["industry_expert"], # 指定执行此任务的Agent
            context=[tasks["message_analysis"]], # 此任务依赖上一个任务的输出
            expected_output="""
详细的产业链分析JSON，重点突出各环节的关键企业和事件影响的传导路径。
确保每个产业链环节都有明确的企业映射和影响分析。
严格按照以下JSON格式输出：
{{
  "target_industry_analysis": {{
    "industry_name": "目标行业名称",
    "industry_overview": "行业概况和特点描述"
  }},
  "detailed_industry_chain": {{
    "upstream_segments": [
      {{
        "segment_name": "环节名称",
        "segment_description": "环节功能描述",
        "technical_barrier": "高/中/低",
        "market_concentration": "高/中/低",
        "key_companies": [
          {{
            "stock_code": "000XXX",
            "company_name": "公司名称",
            "chain_position": "在产业链中的精确位置",
            "market_status": "龙头/重要参与者/新兴企业",
            "core_business": "核心业务描述",
            "competitive_advantage": "竞争优势"
          }}
        ],
        "event_impact_analysis": {{
          "impact_direction": "positive/negative/neutral",
          "impact_mechanism": "具体影响机制说明",
          "impact_intensity": "高/中/低",
          "impact_timing": "即时/短期/中期/长期"
        }}
      }}
    ],
    "midstream_segments": [/*同样格式*/],
    "downstream_segments": [/*同样格式*/]
  }},
  "transmission_path_analysis": {{
    "primary_transmission_paths": [
      {{
        "path_description": "传导路径描述",
        "path_steps": [
          {{
            "step_number": 1,
            "from_element": "起始点",
            "to_element": "传导到的点",
            "transmission_mechanism": "传导机制",
            "transmission_strength": "强/中/弱",
            "time_delay": "传导延迟时间"
          }}
        ],
        "overall_confidence": "路径整体置信度0-1"
      }}
    ]
  }},
  "key_control_points": [
    {{
      "control_point_name": "控制点名称",
      "control_point_type": "技术/供应/需求/政策",
      "importance_reason": "重要性原因",
      "related_companies": ["相关公司列表"]
    }}
  ],
  "chain_summary": {{
    "total_companies_identified": "识别的公司总数",
    "most_impacted_segment": "受影响最大的环节",
    "key_investment_themes": ["投资主题1", "投资主题2"],
    "chain_health_assessment": "产业链健康度评估"
  }},
  "visualization_data": {{
    "industry_chain_network_data": {{
        "nodes": [
            {{"id": "upstream_segment_1_id", "label": "上游环节1", "type": "upstream"}},
            {{"id": "company_A_id", "label": "公司A", "type": "company", "properties": {{"parent_segment": "upstream_segment_1_id"}} }}
        ],
        "edges": [
            {{"source": "upstream_segment_1_id", "target": "midstream_segment_1_id", "label": "供应"}}
        ]
    }},
    "transmission_to_segments_flowchart_data": {{
        "steps": [
             {{"id": "event", "label": "事件核心", "type": "io"}},
             {{"id": "upstream_impact", "label": "对上游环节X的影响", "type": "process"}}
        ],
        "connections": [
            {{"from": "event", "to": "upstream_impact"}}
        ]
    }}
  }}
}}
"""
        )
        
        # 任务3：精细化因果推理与影响评估
        tasks["causal_reasoning"] = Task( # 创建因果推理任务
            description=( # 任务描述
                "结合【消息分析任务】的事件定义和【产业链构建任务】的产业链图谱及股票池：\n"
                "步骤：\n"
                "1. **因果图构建与可视化**：使用【增强型因果图分析工具】，基于事件信息和产业链数据，构建一个详细的、可视化的因果推理图。图中应清晰展示事件如何通过产业链各环节传导影响。\n"
                "2. **传导路径与机制分析**：深入分析事件影响的直接作用机制和间接传导路径，识别出关键的传导节点（可能是特定技术、政策、市场需求变化等）和核心传导路径。\n"
                "3. **宏观环境因素考量**：使用【宏观数据工具】，获取与事件和行业相关的最新宏观经济指标（如PMI、相关政策发布日期、行业景气指数等），评估当前的宏观经济环境对因果传导效果可能产生的增强或削弱作用。\n"
                "4. **影响程度量化评估**：尝试对事件在产业链不同环节、以及对重点关联企业的潜在影响程度进行初步的量化或定性分级评估（例如：强/中/弱正面影响，或强/中/弱负面影响）。"
            ),
            agent=self.agents["causal_expert"], # 指定执行此任务的Agent
            context=[tasks["message_analysis"], tasks["industry_chain_building"]], # 此任务依赖前两个任务的输出
            expected_output=( # 预期的输出格式和内容
                "一个结构化的JSON字符串，必须包含以下字段：\n"
                "  - `causal_graph_analysis`: (object) 详细的因果图分析数据。\n"
                "    - `nodes`: (list of objects) 图中的节点信息（事件、产业链环节、企业等）。\n"
                "    - `edges`: (list of objects) 图中的边信息（传导关系、影响方向、强度初步判断）。\n"
                "  - `visualization_image_base64`: (string) 因果图的Base64编码PNG图像字符串。(此字段由工具直接生成，无需Agent构造)\n"
                "  - `key_transmission_paths_analysis`: (list of strings) 对几条最关键的事件影响传导路径的文字描述和分析。\n"
                "  - `quantitative_impact_assessment`: (list of objects) 对产业链关键环节或代表性企业的初步量化/定性影响评估，每个对象包含评估目标、影响方向、预估强度（强/中/弱）和简要理由。\n"
                "  - `macro_context_impact_summary`: (string) 简要总结当前宏观经济环境对此次事件影响传导的可能作用。\n"
                "  - `visualization_data_for_causal_graph`: {{ /* 用于生成因果图的GraphData结构 */ }}\n"
            )
        )

        # 任务4 (原任务3)：股票池构建任务
        tasks["stock_pool_building"] = Task(
            description="""
基于前面的产业链分析结果，请构建完整的投资标的股票池。

**第一步：股票收集整理**
从产业链分析中提取所有相关股票，并补充遗漏标的：

**数据整理要求：**
- 确认所有股票代码的准确性
- 补充遗漏的重要标的（如果产业链分析中有遗漏）
- 验证公司的主营业务与产业链位置的匹配度
- 剔除停牌、ST、*ST等不可投资标的

**第二步：相关性评估体系**
为每只股票建立多维度相关性评估：

**评估维度：**
1. **业务相关性（40%权重）：**
   - 主营业务占比：该业务在公司总营收中的比例
   - 业务核心度：该业务是否为公司核心战略
   - 评分标准：0.8-1.0（核心业务），0.6-0.8（重要业务），0.4-0.6（一般业务），<0.4（边缘业务）

2. **传导敏感性（30%权重）：**
   - 基于产业链位置评估对事件的敏感程度
   - 上游供应商：中等敏感性
   - 中游制造商：高敏感性  
   - 下游应用商：中等敏感性
   - 评分标准：0.8-1.0（高敏感），0.6-0.8（中敏感），0.4-0.6（低敏感）

3. **市场地位（20%权重）：**
   - 行业龙头：1.0分
   - 重要参与者：0.8分
   - 新兴企业：0.6分
   - 其他参与者：0.4分

4. **财务质量（10%权重）：**
   - 基于盈利能力、成长性、财务健康度
   - 优秀：0.9-1.0，良好：0.7-0.9，一般：0.5-0.7，较差：<0.5

**第三步：股票池分层构建**

**高相关性池（综合评分≥0.75）：**
- 核心投资标的
- 与投资主题高度相关
- 重点推荐和深度分析

**中相关性池（综合评分0.60-0.75）：**
- 重要关注标的
- 有一定投资价值
- 作为组合配置考虑

**低相关性池（综合评分0.45-0.60）：**
- 边缘相关标的
- 弹性配置选择
- 主题拓展考虑

**第四步：投资逻辑构建**
为每只股票构建详细的投资逻辑：

**逻辑要素：**
- 事件传导路径：事件如何传导到该股票
- 业务影响机制：对公司具体业务的影响
- 财务影响预期：对营收、利润的潜在影响
- 催化剂分析：未来可能的积极因素
- 风险因素：需要关注的风险点
""",
            agent=self.agents["stock_pool_builder"],
            context=[tasks["industry_chain_building"]], # 依赖产业链分析结果
            expected_output="""
完整的分层股票池JSON，包含详细的相关性评估和投资逻辑分析。
重点突出高相关性池中股票的深度投资逻辑。
严格按照以下JSON格式输出：
{{
  "stock_pool_construction": {{
    "construction_summary": {{
      "total_stocks": "股票总数",
      "high_relevance_count": "高相关股票数量", 
      "medium_relevance_count": "中相关股票数量",
      "low_relevance_count": "低相关股票数量",
      "construction_date": "构建日期"
    }},
    "high_relevance_pool": [
      {{
        "stock_code": "000XXX",
        "stock_name": "股票名称",
        "industry_chain_position": "在产业链中的位置",
        "relevance_score": 0.XX,
        "score_breakdown": {{
          "business_relevance": 0.XX,
          "transmission_sensitivity": 0.XX, 
          "market_position": 0.XX,
          "financial_quality": 0.XX
        }},
        "investment_logic": {{
          "transmission_path": "事件传导到该股的具体路径",
          "business_impact": "对公司业务的具体影响",
          "financial_impact_expectation": "对财务指标的影响预期",
          "key_catalysts": ["催化剂1", "催化剂2"],
          "main_risks": ["风险点1", "风险点2"]
        }},
        "recommendation_level": "强烈推荐/推荐/关注"
      }}
    ],
    "medium_relevance_pool": [/*同样格式，但投资逻辑可以简化*/],
    "low_relevance_pool": [/*同样格式，投资逻辑进一步简化*/]
  }},
  "pool_analysis": {{
    "sector_distribution": {{
      "upstream_stocks": "上游股票数量",
      "midstream_stocks": "中游股票数量", 
      "downstream_stocks": "下游股票数量"
    }},
    "market_cap_distribution": {{
      "large_cap": "大盘股数量",
      "mid_cap": "中盘股数量",
      "small_cap": "小盘股数量"
    }},
    "risk_level_distribution": {{
      "low_risk": "低风险股票数量",
      "medium_risk": "中风险股票数量",
      "high_risk": "高风险股票数量"
    }}
  }},
  "investment_themes": [
    {{
      "theme_name": "投资主题名称",
      "theme_description": "主题描述",
      "related_stocks": ["相关股票代码"],
      "theme_strength": "主题强度评估"
    }}
  ],
  "visualization_data": {{
      "stock_pool_treemap_data": {{
          "id": "root_pool", "name": "总股票池", "children": [
              {{"id": "high_rel_node", "name": "高相关", "children": [ {{"id": "stock1_id", "name": "股票1", "value": 1}} ] }}
          ]
      }}
  }}
}}
"""
        )
        
        # 任务5 (原任务4)：智能化投资组合构建与建议
        tasks["investment_advice"] = Task( # 创建投资建议任务
            description="""
基于【股票池构建任务】输出的分层股票池和【因果推理任务】的深度分析结果：

**第一步：核心标的筛选**
从"高相关性池"和"中相关性池"中，结合因果推理的结论（如影响强度、时间窗口、关键传导路径），筛选出3-5只最具投资价值的核心标的。

**筛选标准：**
- 传导逻辑清晰且强度高
- 公司基本面优质
- 市场地位稳固或成长性突出
- 估值相对合理
- 事件催化剂明确

**第二步：深度投资逻辑阐述**
为每只核心标的构建详细的、可追溯的投资逻辑：
- **完整传导链条**：清晰展示"新闻事件 → A股行业板块 → 产业链环节 → 具体公司业务 → 财务预期 → 投资决策"的完整逻辑。
- **核心驱动因素**：明确指出推动投资价值的核心驱动力。
- **量化指标支撑**：尽可能使用量化数据（如预期营收增长、利润弹性）支撑逻辑。
- **情景分析**：简要分析在不同市场情景下的表现预期。

**第三步：投资组合构建**
- 基于核心标的，构建一个或多个示例投资组合。
- 考虑组合的风险分散和收益目标。
- 明确组合的投资主题和适用投资者类型。

**第四步：风险全面揭示**
- 针对每个核心标的和整体组合，进行全面的风险分析。
- 包括市场风险、行业风险、公司特有风险、事件发展不及预期风险等。
- 提供风险应对建议。

**第五步：操作策略建议**
- 给出具体的操作建议，如买入时机、仓位配置、止盈止损策略等。
- 强调投资的时间周期和预期回报。
""", # 任务描述
            agent=self.agents["investment_advisor"], # 指定执行此任务的Agent
            context=[tasks["causal_reasoning"], tasks["stock_pool_building"]], # 依赖因果推理和股票池构建结果
            expected_output="""
一个结构化的JSON字符串，必须包含以下字段：
{{
  "investment_strategy_summary": {{
    "core_investment_theme": "本次投资策略的核心主题",
    "strategy_overview": "投资策略概述（如事件驱动、价值发现等）",
    "target_investor_profile": "目标投资者画像（风险偏好、投资周期）",
    "expected_return_range": "预期回报区间（年化）",
    "recommended_investment_horizon": "建议投资周期（如6-12个月）"
  }},
  "core_stock_recommendations": [
    {{
      "stock_code": "000XXX",
      "stock_name": "股票名称",
      "recommendation_rationale": {{
        "full_transmission_logic": "详细的'新闻→行业→产业链→公司→决策'逻辑链",
        "key_investment_drivers": ["核心驱动因素1", "核心驱动因素2"],
        "quantitative_support": [
          {{"metric_name": "营收增长预期", "value": "15-20%"}},
          {{"metric_name": "利润弹性", "value": "高"}}
        ],
        "scenario_analysis_summary": "不同情景下的简要表现预期"
      }},
      "operational_advice": {{
        "entry_point_suggestion": "建议买入时机或价格区间",
        "position_sizing_recommendation": "建议仓位配置（如占总投资的X%）",
        "stop_loss_suggestion": "止损建议",
        "target_price_range": "目标价格区间"
      }},
      "risk_assessment": {{
        "specific_risks": ["个股特有风险1", "个股特有风险2"],
        "risk_mitigation_suggestions": ["风险应对建议1"]
      }}
    }}
  ],
  "sample_portfolio_construction": {{
    "portfolio_name": "示例组合名称（如：新能源政策驱动组合）",
    "portfolio_description": "组合构建逻辑和目标",
    "asset_allocation": [
      {{"stock_code": "000XXX", "weight": "40%"}},
      {{"stock_code": "000YYY", "weight": "30%"}},
      {{"stock_code": "000ZZZ", "weight": "30%"}}
    ],
    "portfolio_risk_level": "组合风险等级（高/中/低）",
    "expected_portfolio_volatility": "组合预期波动率"
  }},
  "overall_strategy_risks": [
    {{
      "risk_category": "整体策略风险类别（如市场系统性风险）",
      "risk_details": "风险详细描述",
      "contingency_plan_summary": "应对预案摘要"
    }}
  ],
  "disclaimer": "投资有风险，入市需谨慎。本建议仅供参考，不构成任何投资承诺。",
  "visualization_data": {{
      "final_decision_flowchart_data": {{
          "steps": [
              {{"id": "stock_pool", "label": "分层股票池", "type": "io"}},
              {{"id": "screening", "label": "核心标的筛选", "type": "process"}},
              {{"id": "final_reco", "label": "最终推荐", "type": "output"}}
          ],
          "connections": [
              {{"from": "stock_pool", "to": "screening"}},
              {{"from": "screening", "to": "final_reco"}}
          ]
      }}
  }}
}}
"""
        )
        
        self.logger.info(f"成功创建 {len(tasks)} 个Agent任务") # 记录日志：成功创建任务的数量
        return tasks # 返回创建的任务字典

    def _generate_agent_summary(self, agent_result: Any) -> Dict[str, Any]:
        """
        为Agent团队的分析结果生成增强型摘要。
        Generate enhanced summary for Agent crew analysis result.
        
        Args:
            agent_result: Agent团队的最终输出 (通常是最后一个Task的输出字典)。
                          Final output from Agent crew.
            
        Returns:
            Dict[str, Any]: 增强型摘要信息。
                          Enhanced summary information.
        """
        # 初始化增强型摘要
        enhanced_summary = {
            "transmission_overview": {
                "news_to_industry": "新闻事件对相关A股行业板块的影响分析正在生成...",
                "industry_to_chain": "受影响行业板块内部的产业链传导分析正在生成...",
                "chain_to_stocks": "产业链环节到相关股票池的映射分析正在生成...",
                "final_logic": "最终投资逻辑和核心推荐总结正在生成..."
            },
            "layered_results": {
                "affected_industries": [], # 将从早期任务中提取
                "key_chain_segments": [],  # 将从产业链分析任务中提取
                "stock_pool_summary": "股票池构建和分层统计正在生成...",
                "final_recommendations": [] # 将从投资决策任务中提取
            },
            "confidence_analysis": {
                "overall_confidence": 0.0, # 将综合计算
                "key_assumptions": [], # 将从各阶段分析中提取
                "risk_factors": [] # 将从各阶段分析中提取
            },
            # 保留main.py期望的旧版摘要字段，并尝试填充
            "message": "Agent团队分析完成",
            "event_type": "未知", # 尝试从agent_result 或早期任务中获取
            "impact_direction": "未知", # 尝试从agent_result 或早期任务中获取
            "recommended_stocks_count": 0, # 将从final_recommendations计算
            "top_recommendation": None, # 将从final_recommendations计算
            "investment_theme": "未明确投资主题" # 将从agent_result获取
        }

        try:
            if isinstance(agent_result, dict):
                # 填充投资主题
                enhanced_summary["investment_theme"] = agent_result.get("investment_strategy_summary", {}).get("core_investment_theme", "未明确投资主题")

                # 填充最终推荐和相关统计
                core_recommendations = agent_result.get("core_stock_recommendations", [])
                enhanced_summary["layered_results"]["final_recommendations"] = core_recommendations
                enhanced_summary["recommended_stocks_count"] = len(core_recommendations)
                if core_recommendations:
                    top_stock = core_recommendations[0]
                    enhanced_summary["top_recommendation"] = {
                        "name": top_stock.get("stock_name", "未知股票"),
                        "code": top_stock.get("stock_code", "N/A")
                    }
                    # 尝试填充旧版摘要的 event_type, affected_industries, impact_direction
                    # 这些信息理想情况下应从更早的任务输出中获取并传递到最终结果
                    # 这里我们尝试从最终推荐的逻辑中反向推断一些信息，但这并不完美
                    if top_stock.get("recommendation_rationale", {}).get("full_transmission_logic"):
                        logic_parts = top_stock["recommendation_rationale"]["full_transmission_logic"].split('→')
                        if len(logic_parts) > 1: # 假设第一个是事件/新闻相关
                             # 这是一个非常粗略的提取，实际应从前置任务获取
                            enhanced_summary["event_type"] = "根据新闻分析" 
                        if len(logic_parts) > 2: # 假设第二个是行业相关
                            enhanced_summary["layered_results"]["affected_industries"] = [logic_parts[1].strip()]


                # 填充股票池摘要 (这部分数据应来自股票池构建Agent的输出)
                # 由于当前agent_result是投资决策Agent的输出，我们只能做一些推断
                # 理想情况下，run_analysis_crew应该聚合所有任务的输出
                stock_pool_construction = agent_result.get("stock_pool_construction", {}) # 假设投资决策Agent的输出包含了这个
                if stock_pool_construction:
                     enhanced_summary["layered_results"]["stock_pool_summary"] = (
                        f"高相关: {stock_pool_construction.get('construction_summary',{}).get('high_relevance_count',0)}只, "
                        f"中相关: {stock_pool_construction.get('construction_summary',{}).get('medium_relevance_count',0)}只, "
                        f"低相关: {stock_pool_construction.get('construction_summary',{}).get('low_relevance_count',0)}只"
                    )
                
                # 填充置信度、假设和风险 (这些也应从各阶段聚合)
                overall_risks = agent_result.get("overall_strategy_risks", [])
                if overall_risks:
                    enhanced_summary["confidence_analysis"]["risk_factors"] = [
                        f"{r.get('risk_category')}: {r.get('risk_details')}" for r in overall_risks
                    ]
                
                # 简化的置信度计算 - 实际应更复杂
                if core_recommendations:
                    enhanced_summary["confidence_analysis"]["overall_confidence"] = 0.75 # 示例值
                
                # 传导总览的填充需要访问所有中间任务的结果，这里仅作占位
                # enhanced_summary["transmission_overview"]["news_to_industry"] = "具体分析见各层级详情"
                # ... 其他传导总览字段

            else:
                enhanced_summary["message"] = "Agent团队最终输出格式非预期字典类型"
                self.logger.warning(f"Agent团队最终输出格式非预期: {type(agent_result)}")

        except Exception as e:
            self.logger.error(f"生成增强型Agent摘要时发生错误: {e}", exc_info=True)
            enhanced_summary["message"] = f"生成摘要时出错: {str(e)}"
            # 确保即使出错，核心摘要字段也以默认值存在
            enhanced_summary.setdefault("event_type", "未知")
            enhanced_summary.setdefault("layered_results", {}).setdefault("affected_industries", [])
            enhanced_summary.setdefault("impact_direction", "未知")
            enhanced_summary.setdefault("recommended_stocks_count", 0)
            enhanced_summary.setdefault("top_recommendation", None)
            enhanced_summary.setdefault("investment_theme", "处理摘要时出错")
            
        return enhanced_summary

if __name__ == "__main__": # 如果此文件作为主程序运行
    # 测试Agent协作管理器 - Test Agent Collaboration Manager
    print("测试Agent协作管理器...") # 打印测试信息
    
    # 创建LLM客户端和Agent管理器 - Create LLM client and Agent manager
    llm_client = LLMClient( # 创建LLM客户端实例
        base_url=API_CONFIG["base_url"], # API基础URL
        api_key=API_CONFIG["api_key"], # API密钥
        model=API_CONFIG.get("model", "deepseek-chat") # 模型名称
    )
    crew_manager = CrewManager(llm_client) # 创建CrewManager实例
    
    # 测试消息 - Test message
    test_message = """
    国家发改委今日发布《关于促进集成电路产业高质量发展的若干政策》，
    提出加大财税支持力度，优化融资环境，加强人才培养。
    预计将对国内半导体产业链产生深远影响。
    """ # 定义测试消息文本
    
    # 运行Agent团队分析 - Run Agent crew analysis
    try:
        # 实际调用 run_analysis_crew
        print(f"开始使用真实消息进行分析: {test_message[:50]}...")
        result = crew_manager.run_analysis_crew(test_message) # 运行分析
        
        # 打印完整结果，方便调试
        print("\n" + "="*20 + " 完整分析结果 " + "="*20)
        print(json.dumps(result, ensure_ascii=False, indent=2)) # 打印格式化的JSON结果
        print("="*50 + "\n")

        if result.get("success"): # 检查分析是否成功
            print("✓ Agent团队分析成功") # 打印成功信息
            summary = result.get("summary", {}) # 获取摘要信息
            print(f"分析摘要: {summary}") # 打印摘要
            print(f"用量统计: {result.get('crew_execution_stats', {})}") # 打印用量统计
        else: # 如果分析失败
            print(f"✗ Agent团队分析失败: {result.get('error_details', '未知错误')}") # 打印失败信息
            
    except Exception as e: # 捕获测试过程中的异常
        print(f"✗ 测试执行过程中发生严重异常: {e}") # 打印异常信息
        import traceback # 导入traceback模块
        traceback.print_exc() # 打印详细的异常堆栈信息
    
    print("测试完成") # 打印测试完成信息
