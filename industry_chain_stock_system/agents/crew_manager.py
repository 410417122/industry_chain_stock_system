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
            role=msg_analyst_config.get("role", "资深财经新闻分析师"), # Agent角色
            goal=( # Agent目标，进行详细描述
                "精准、全面地解析财经消息，提取关键事件信息、影响范围、情感倾向，"
                "并利用搜索工具补充事件背景和相关信息，确保分析的深度和广度。"
            ),
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
            role=industry_expert_config.get("role", "产业链高级研究员"), # Agent角色
            goal=( # Agent目标，进行详细描述
                "基于事件信息和深入的行业研究，构建完整、准确、动态更新的产业链知识图谱。"
                "识别核心影响环节、关键技术节点和代表性企业，并清晰梳理产业链上下游的传导关系。"
            ),
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
        
        # 投资顾问Agent - Investment Advisor Agent
        investment_advisor_config = AGENT_CONFIG.get("investment_advisor", {}) # 从配置中获取投资顾问Agent的配置
        agents["investment_advisor"] = Agent( # 创建投资顾问Agent实例
            role=investment_advisor_config.get("role", "资深量化投资策略师"), # Agent角色
            goal=( # Agent目标，进行详细描述
                "基于深度因果分析和产业链研究成果，运用量化模型筛选构建高相关性的股票池。"
                "对候选股票进行精细化的相关性评估（强、中、弱），并结合基本面和技术面分析，"
                "最终提供具体、可操作、风险可控的投资组合建议，并附带详尽的投资逻辑和风险收益分析。"
            ),
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
            description=( # 任务描述，详细说明任务目标和步骤
                f"深入分析以下财经消息：'{message_text}'。\n"
                "步骤：\n"
                "1. **核心事件提取**：使用【消息分析工具】精准提取消息中的关键事件信息，包括：事件类型（如政策发布、技术突破、市场异动等）、核心主体（涉及的公司、机构、产品等）、主要影响领域、以及消息中体现的初步情感倾向（正面/负面/中性）。\n"
                "2. **背景信息挖掘**：利用【互联网搜索工具】，针对提取出的核心事件和主体，进行深入的背景信息检索。重点查找与事件相关的历史渊源、深度解读文章、权威机构评论以及当前市场的普遍反应。\n"
                "3. **信息整合与事件定义**：综合以上两步获取的所有信息，形成对该财经事件全面、准确、多维度的定义和理解。确保输出结果清晰、结构化，便于后续任务使用。"
            ),
            agent=self.agents["message_analyst"], # 指定执行此任务的Agent
            expected_output=( # 预期的输出格式和内容
                "一个结构化的JSON字符串，必须包含以下字段：\n"
                "  - `event_details`: (object) 包含从消息中直接提取的详细事件信息。\n"
                "    - `type`: (string) 事件类型（如：'政策调整', '技术突破', '市场传闻'），如果无法确定，请明确指出“未知”。\n"
                "    - `main_subjects`: (list of strings) 事件涉及的主要公司、机构或产品名称。\n"
                "    - `scope_of_impact`: (string) 初步判断的事件主要影响范围或领域。\n"
                "    - `initial_sentiment`: (string) 基于消息文本的初步情感判断（'正面', '负面', '中性'）。\n"
                "  - `background_research`: (object) 通过互联网搜索补充的背景信息。\n"
                "    - `key_findings`: (list of strings) 最重要的几条背景信息或深度解读摘要。\n"
                "    - `market_reactions_summary`: (string) 对当前市场主要评论和反应的总结。\n"
                "  - `refined_event_definition`: (string) 综合所有信息后，对事件的最终精准定义描述。\n"
                "  - `preliminary_affected_industries`: (list of strings) 根据分析，列出1-3个最可能受此事件直接影响的行业名称。如果无法准确判断，请返回一个空列表 `[]` 或包含如“不明确”等说明性条目的列表，但此字段必须存在。"
            )
        )
        
        # 任务2：产业链深度分析与构建
        tasks["industry_chain_building"] = Task( # 创建产业链构建任务
            description=( # 任务描述
                "基于【消息分析任务】输出的`preliminary_affected_industries`初步判断的受影响行业列表：\n"
                "步骤：\n"
                "1. **核心行业确认与产业链研究**：从初步影响行业列表中，筛选并确认1-2个核心受影响最直接的行业。针对这些核心行业，使用【行业研究工具】（例如，`search_type`可以设为`'overview'`或`'chain'`）进行深度搜索，全面了解其当前的产业链结构，包括明确的上游（原材料/零部件供应）、中游（核心制造/加工/服务）和下游（应用/分销/消费）环节。\n"
                "2. **产业链股票池构建**：使用【行业股票工具】获取核心行业及其已识别的上下游相关行业的完整股票列表。调用此工具时，请注意参数：`industry`应为行业名称或代码；`classification`参数应设为`'sw'` (代表申万行业分类) 或 `'zjh'` (代表证监会行业分类)；`detail_level`参数应设为字符串`'1'`、`'2'`或`'3'`，分别代表一、二、三级行业。例如，要获取申万一级行业的股票，可以设置 `classification='sw'` 和 `detail_level='1'`。\n"
                "3. **股票基础信息收集**：针对股票池中的每只股票，使用【股票数据工具】（例如，`data_type`可以设为`'real_time'`获取实时行情或`'fundamental'`获取基本面信息）获取其关键基础信息，至少包括：股票代码、股票简称、总市值、流通市值、主营业务构成等。\n"
                "4. **产业链知识图谱构建**：整合以上信息，构建一个详细的、结构化的产业链知识图谱。图谱应清晰展示核心行业、上下游环节、各环节的代表性企业（股票）及其主要的业务联系或供应关系。"
            ),
            agent=self.agents["industry_expert"], # 指定执行此任务的Agent
            context=[tasks["message_analysis"]], # 此任务依赖上一个任务的输出
            expected_output=( # 预期的输出格式和内容
                "一个结构化的JSON字符串，必须包含以下字段：\n"
                "  - `target_core_industry`: (string) 最终确认的核心受影响行业名称。\n"
                "  - `industry_chain_map`: (object) 详细的产业链图谱描述。\n"
                "    - `upstream`: (object) 上游环节描述，包含环节名称和代表性企业列表（含股票代码、简称）。\n"
                "    - `midstream`: (object) 中游环节描述，包含环节名称和代表性企业列表。\n"
                "    - `downstream`: (object) 下游环节描述，包含环节名称和代表性企业列表。\n"
                "  - `full_industry_stock_pool`: (list of objects) 完整的产业链相关股票池，每个对象包含股票代码、简称、所属产业链环节、主营业务简介。"
            )
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
                "  - `visualization_image_base64`: (string) 因果图的Base64编码PNG图像字符串。\n"
                "  - `key_transmission_paths_analysis`: (list of strings) 对几条最关键的事件影响传导路径的文字描述和分析。\n"
                "  - `quantitative_impact_assessment`: (list of objects) 对产业链关键环节或代表性企业的初步量化/定性影响评估，每个对象包含评估目标、影响方向、预估强度（强/中/弱）和简要理由。\n"
                "  - `macro_context_impact_summary`: (string) 简要总结当前宏观经济环境对此次事件影响传导的可能作用。"
            )
        )
        
        # 任务4：智能化投资组合构建与建议
        tasks["investment_advice"] = Task( # 创建投资建议任务
            description=( # 任务描述
                "基于【因果推理任务】输出的深度分析结果和【产业链构建任务】的股票池：\n"
                "步骤：\n"
                "1. **初步股票筛选**：使用【股票筛选工具】，结合因果分析中的`quantitative_impact_assessment`（量化影响评估），从`full_industry_stock_pool`（产业链股票池）中，筛选出受事件积极影响（或在特定策略下，受负面影响可做空）的初步候选股票列表。\n"
                "2. **相关性深度评估**：使用【相关性分析工具】，对初步候选股票列表中的每只股票，进行与核心事件的强、中、弱相关性分类。分析应综合考虑业务直接关联度、产业链位置、事件关键词匹配度等因素。\n"
                "3. **核心股票深度画像**：对于被评估为“强相关”和“中相关”的股票，使用【股票数据工具】获取其最新的、详细的基本面数据（如财务指标、估值、成长性）和近期行情数据。\n"
                "4. **投资组合构建与建议**：综合股票的基本面、技术面趋势（如有数据）、事件驱动的因果影响强度、以及与事件的相关性等级，最终形成结构化的投资建议。建议应包括：\n"
                "    - 明确的“强相关”股票池，并针对其中1-3只核心标的给出具体的操作建议（如：买入、增持、重点关注）及详细的投资逻辑。\n"
                "    - “中相关”股票池，并说明其值得关注的理由。\n"
                "    - “弱相关”股票池，作为参考简要列出。\n"
                "5. **风险揭示**：对提出的投资建议，必须进行全面的潜在风险分析和提示。"
            ),
            agent=self.agents["investment_advisor"], # 指定执行此任务的Agent
            context=[tasks["causal_reasoning"], tasks["industry_chain_building"]], # 此任务依赖前两个任务的输出
            expected_output=( # 预期的输出格式和内容
                "一个结构化的JSON字符串，必须包含以下字段：\n"
                "  - `investment_theme_summary`: (string) 本次投资建议的核心主题和关键投资逻辑概述。\n"
                "  - `strongly_related_portfolio`: (object) 强相关股票组合建议。\n"
                "    - `count`: (integer) 强相关股票数量。\n"
                "    - `stocks`: (list of objects) 每只强相关股票的详细信息，包含：\n"
                "      - `code`: (string) 股票代码。\n"
                "      - `name`: (string) 股票名称。\n"
                "      - `relevance_score`: (float) 相关性评分。\n"
                "      - `current_price`: (float, optional) 最新股价（如果能获取）。\n"
                "      - `investment_logic`: (string) 详细的投资逻辑阐述。\n"
                "      - `specific_advice`: (string) 具体操作建议（如：'建议重点关注并逢低布局'）。\n"
                "      - `potential_catalysts`: (list of strings) 未来可能的催化剂。\n"
                "      - `key_risks`: (list of strings) 主要风险点提示。\n"
                "  - `moderately_related_watchlist`: (object) 中相关股票观察池。\n"
                "    - `count`: (integer) 中相关股票数量。\n"
                "    - `stocks`: (list of objects) 每只中相关股票信息，包含代码、名称、关注理由。\n"
                "  - `weakly_related_reference`: (object, optional) 弱相关股票参考列表（可选，如果数量过多可省略）。\n"
                "    - `count`: (integer) 弱相关股票数量。\n"
                "    - `stocks`: (list of objects) 每只弱相关股票信息，包含代码、名称。\n"
                "  - `overall_portfolio_risk_assessment`: (string) 对整个推荐组合的综合风险评估和管理建议。"
            )
        )
        
        self.logger.info(f"成功创建 {len(tasks)} 个Agent任务") # 记录日志：成功创建任务的数量
        return tasks # 返回创建的任务字典

    def _generate_agent_summary(self, agent_result: Any) -> Dict[str, Any]:
        """
        为Agent团队的分析结果生成摘要。
        Generate summary for Agent crew analysis result.
        
        Args:
            agent_result: Agent团队的最终输出 (通常是最后一个Task的输出字典)。
                          Final output from Agent crew.
            
        Returns:
            Dict[str, Any]: 摘要信息。
                          Summary information.
        """
        # 初始化摘要，包含 main.py 中期望的字段的默认值
        summary = {
            "message": "Agent团队分析完成",
            "event_type": "未知",
            "affected_industries": [],
            "impact_direction": "未知",
            "recommended_stocks_count": 0,
            "top_recommendation": None, # 期望是一个包含 name 和 code 的字典
            "investment_theme": "未明确投资主题" # 保留原有字段
        }
        
        try:
            if isinstance(agent_result, dict):
                summary["investment_theme"] = agent_result.get("investment_theme_summary", "未明确投资主题")

                strong_portfolio = agent_result.get("strongly_related_portfolio", {})
                strong_stocks = strong_portfolio.get("stocks", [])
                
                medium_portfolio = agent_result.get("moderately_related_watchlist", {})
                # medium_stocks_count = medium_portfolio.get("count", 0) # 'count' 字段可能不存在或不准确
                medium_stocks_list = medium_portfolio.get("stocks", [])
                medium_stocks_count = len(medium_stocks_list)


                summary["recommended_strong_stocks_count"] = len(strong_stocks) # 单独记录强相关数量
                summary["recommended_medium_stocks_count"] = medium_stocks_count # 单独记录中相关数量
                
                total_recommended_count = len(strong_stocks) + medium_stocks_count
                summary["recommended_stocks_count"] = total_recommended_count # 总推荐数量

                if strong_stocks:
                    top_stock_data = strong_stocks[0]
                    summary["top_recommendation"] = {
                        "name": top_stock_data.get("name", "未知股票"),
                        "code": top_stock_data.get("code", "N/A"),
                        # "advice": top_stock_data.get("specific_advice", "未提供具体建议") # main.py的摘要不直接用此字段
                    }
                
                # 注意: event_type, affected_industries, impact_direction 
                # 通常在投资顾问Agent的最终输出中不直接包含。
                # 这些信息更可能来自早期的分析阶段 (如消息分析Agent)。
                # 要在最终摘要中包含它们，需要调整Crew的数据流，
                # 例如，让投资顾问在其输出中包含这些上下文，
                # 或者修改 run_analysis_crew 方法以聚合来自不同任务的输出。
                # 当前修改仅基于 agent_result (投资顾问的输出) 进行提取。

            elif isinstance(agent_result, list) and agent_result:
                # 旧的兼容逻辑，如果最终结果直接是一个列表（不太可能了）
                summary["recommended_stocks_count"] = len(agent_result)
                if agent_result[0] and isinstance(agent_result[0], dict):
                    summary["top_recommendation"] = {
                        "name": agent_result[0].get("stock_name", agent_result[0].get("name", "未知股票")),
                        "code": agent_result[0].get("stock_code", agent_result[0].get("code", "N/A")),
                    }
            else:
                summary["error_in_summary_generation"] = "最终结果格式不符合预期，无法提取标准摘要信息。"
                
        except Exception as e:
            self.logger.error(f"生成Agent摘要时发生错误: {e}", exc_info=True)
            summary["error_in_summary_generation"] = f"生成摘要时出错: {str(e)}"
            # 确保即使出错，核心摘要字段也以默认值存在
            summary.setdefault("event_type", "未知")
            summary.setdefault("affected_industries", [])
            summary.setdefault("impact_direction", "未知")
            summary.setdefault("recommended_stocks_count", 0)
            summary.setdefault("top_recommendation", None)
            summary.setdefault("investment_theme", "处理摘要时出错")
            
        return summary

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
