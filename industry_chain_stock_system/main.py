"""
消息驱动的产业链选股系统 - 主入口文件
Industry Chain Stock Selection System - Main Entry Point

此文件作为系统的主入口，提供命令行界面和基本的系统运行功能
This file serves as the main entry point for the system, providing CLI and basic system functionality
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径 - Add project root directory to Python path
sys.path.append(str(Path(__file__).parent))

# 导入配置文件 - Import configuration
from config import API_CONFIG, SYSTEM_CONFIG, AGENT_CONFIG

# 导入核心模块 - Import core modules
from modules.message_processor import MessageProcessor
from modules.industry_chain import IndustryChainBuilder
from modules.causal_reasoning import CausalReasoningEngine
from modules.investment_decision import InvestmentAdvisor

# 导入Agent模块 - Import Agent modules
from agents.crew_manager import CrewManager

# 导入工具函数 - Import utility functions
from utils.logger import setup_logger
from utils.api_client import LLMClient

class IndustryChainStockSystem:
    """
    消息驱动的产业链选股系统主类
    Main class for the Industry Chain Stock Selection System
    """
    
    def __init__(self):
        """初始化系统 - Initialize system"""
        # 设置日志 - Setup logging
        self.logger = setup_logger()
        self.logger.info("初始化消息驱动的产业链选股系统...")
        
        # 初始化LLM客户端 - Initialize LLM client
        self.llm_client = LLMClient(
            base_url=API_CONFIG["base_url"],
            api_key=API_CONFIG["api_key"],
            model=API_CONFIG["model"]
        )
        
        # 初始化核心模块 - Initialize core modules
        self.message_processor = MessageProcessor(self.llm_client)
        self.industry_chain_builder = IndustryChainBuilder(self.llm_client)
        self.causal_reasoning_engine = CausalReasoningEngine(self.llm_client)
        self.investment_advisor = InvestmentAdvisor(self.llm_client)
        
        # 初始化Agent管理器 - Initialize Agent manager
        self.crew_manager = CrewManager(self.llm_client)
        
        self.logger.info("系统初始化完成")
        
    def analyze_message(self, message_text: str, mode: str = "full") -> dict:
        """
        分析财经消息并生成投资建议
        Analyze financial message and generate investment advice
        
        Args:
            message_text: 财经消息文本 - Financial message text
            mode: 分析模式 - Analysis mode ("quick", "standard", "full")
            
        Returns:
            dict: 分析结果 - Analysis results
        """
        self.logger.info(f"开始分析消息（模式：{mode}）...")
        
        try:
            if mode == "agent":
                # 使用Agent协作模式 - Use Agent collaboration mode
                result = self.crew_manager.run_analysis_crew(message_text) # 修正方法名称
            else:
                # 使用传统模块化模式 - Use traditional modular mode
                result = self._traditional_analysis(message_text, mode)
            
            self.logger.info("消息分析完成")
            return result
            
        except Exception as e:
            self.logger.error(f"分析过程中出现错误: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _traditional_analysis(self, message_text: str, mode: str) -> dict:
        """
        传统模块化分析流程
        Traditional modular analysis workflow
        """
        # 第1步：消息处理 - Step 1: Message processing
        self.logger.info("步骤1: 处理消息...")
        event_info = self.message_processor.parse_message(message_text)
        
        # 第2步：产业链构建 - Step 2: Industry chain building
        if event_info and event_info.get("affected_industries"):
            self.logger.info("步骤2: 构建产业链...")
            industry_code = event_info["affected_industries"][0]  # 取第一个相关行业
            industry_chain = self.industry_chain_builder.build_industry_chain(industry_code)
        else:
            industry_chain = {}
        
        # 第3步：因果推理 - Step 3: Causal reasoning
        if event_info and industry_chain:
            self.logger.info("步骤3: 因果推理分析...")
            causal_graph = self.causal_reasoning_engine.create_causal_graph(
                event_info, industry_chain
            )
        else:
            causal_graph = {}
        
        # 第4步：投资建议 - Step 4: Investment advice
        if causal_graph:
            self.logger.info("步骤4: 生成投资建议...")
            recommendations = self.investment_advisor.generate_recommendations(
                causal_graph, industry_chain
            )
        else:
            recommendations = []
        
        # 构建最终结果 - Build final result
        result = {
            "success": True,
            "analysis_mode": mode,
            "timestamp": datetime.now().isoformat(),
            "input_message": message_text,
            "event_info": event_info,
            "industry_chain": industry_chain,
            "causal_analysis": causal_graph,
            "investment_recommendations": recommendations,
            "summary": self._generate_summary(event_info, recommendations)
        }
        
        return result
    
    def _generate_summary(self, event_info: dict, recommendations: list) -> dict:
        """
        生成分析摘要
        Generate analysis summary
        """
        if not event_info or not recommendations:
            return {"message": "分析数据不完整，无法生成摘要"}
        
        return {
            "event_type": event_info.get("event_type", "未知"),
            "affected_industries": event_info.get("affected_industries", []),
            "impact_direction": event_info.get("impact_direction", "未知"),
            "recommended_stocks_count": len(recommendations),
            "top_recommendation": recommendations[0] if recommendations else None,
            "confidence_level": event_info.get("confidence", "未知")
        }
    
    def save_result(self, result: dict, filename: str = None) -> str:
        """
        保存分析结果到文件
        Save analysis result to file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/analysis_results/analysis_{timestamp}.json"
        
        # 确保目录存在 - Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"分析结果已保存到: {filename}")
        return filename

def create_cli_parser():
    """创建命令行参数解析器 - Create CLI argument parser"""
    parser = argparse.ArgumentParser(
        description="消息驱动的产业链选股系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例 - Usage Examples:
  python main.py --message "新能源汽车补贴政策调整" --mode agent
  python main.py --file news.txt --output result.json
  python main.py --interactive
        """
    )
    
    # 输入参数 - Input parameters
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--message", "-m",
        type=str,
        help="财经消息文本"
    )
    input_group.add_argument(
        "--file", "-f",
        type=str,
        help="包含财经消息的文件路径"
    )
    input_group.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="启动交互式模式"
    )
    
    # 分析参数 - Analysis parameters
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "full", "agent"],
        default="standard",
        help="分析模式（默认：standard）"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出文件路径"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细输出"
    )
    
    return parser

def interactive_mode(system):
    """交互式模式 - Interactive mode"""
    print("\n=== 消息驱动的产业链选股系统 ===")
    print("输入财经消息，系统将为您分析产业链影响并提供投资建议")
    print("输入 'quit' 或 'exit' 退出程序\n")
    
    while True:
        try:
            # 获取用户输入 - Get user input
            message = input("请输入财经消息: ").strip()
            
            if message.lower() in ['quit', 'exit', '退出']:
                print("感谢使用，再见！")
                break
            
            if not message:
                print("请输入有效的消息内容")
                continue
            
            # 获取分析模式 - Get analysis mode
            mode = input("选择分析模式 [quick/standard/full/agent] (默认: agent): ").strip()
            if not mode:
                mode = "agent"
            
            print(f"\n正在分析消息: {message[:50]}...")
            
            # 执行分析 - Execute analysis
            result = system.analyze_message(message, mode)
            
            # 显示结果 - Display results
            if result.get("success"):
                print("\n=== 分析结果 ===")
                summary = result.get("summary", {})
                print(f"事件类型: {summary.get('event_type', '未知')}")
                print(f"影响行业: {', '.join(summary.get('affected_industries', []))}")
                print(f"影响方向: {summary.get('impact_direction', '未知')}")
                print(f"推荐股票数量: {summary.get('recommended_stocks_count', 0)}")
                
                if summary.get('top_recommendation'):
                    top = summary['top_recommendation']
                    print(f"首选推荐: {top.get('name', '未知')} ({top.get('code', 'N/A')})")
                
                # 询问是否保存结果 - Ask if save results
                save = input("\n是否保存详细结果到文件? [y/N]: ").strip().lower()
                if save in ['y', 'yes', '是']:
                    filename = system.save_result(result)
                    print(f"结果已保存到: {filename}")
            else:
                print(f"分析失败: {result.get('error', '未知错误')}")
            
            print("\n" + "="*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，正在退出...")
            break
        except Exception as e:
            print(f"发生错误: {e}")

def main():
    """主函数 - Main function"""
    # 解析命令行参数 - Parse command line arguments
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # 初始化系统 - Initialize system
    try:
        system = IndustryChainStockSystem()
    except Exception as e:
        print(f"系统初始化失败: {e}")
        sys.exit(1)
    
    # 执行相应的模式 - Execute corresponding mode
    if args.interactive:
        # 交互式模式 - Interactive mode
        interactive_mode(system)
    
    elif args.message:
        # 单条消息分析 - Single message analysis
        print(f"分析消息: {args.message}")
        result = system.analyze_message(args.message, args.mode)
        
        # 输出结果 - Output results
        if args.output:
            system.save_result(result, args.output)
            print(f"结果已保存到: {args.output}")
        else:
            if args.verbose:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                summary = result.get("summary", {})
                print(f"分析完成 - 推荐{summary.get('recommended_stocks_count', 0)}只股票")
    
    elif args.file:
        # 文件分析 - File analysis
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                message = f.read().strip()
            
            print(f"从文件读取消息: {args.file}")
            result = system.analyze_message(message, args.mode)
            
            # 输出结果 - Output results
            if args.output:
                system.save_result(result, args.output)
                print(f"结果已保存到: {args.output}")
            else:
                summary = result.get("summary", {})
                print(f"分析完成 - 推荐{summary.get('recommended_stocks_count', 0)}只股票")
                
        except FileNotFoundError:
            print(f"文件不存在: {args.file}")
            sys.exit(1)
        except Exception as e:
            print(f"读取文件失败: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
