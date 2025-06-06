"""
因果分析工具模块
Causal Analysis Tools Module

包含用于深度因果链分析的增强型Agent工具
Contains enhanced Agent tools for deep causal chain analysis
"""

import json # 导入json库，用于处理JSON数据
from crewai.tools import BaseTool # 从crewai.tools导入BaseTool，用于创建自定义工具
from pydantic import ConfigDict # 导入ConfigDict用于模型配置
from pathlib import Path # 导入Path，用于处理文件路径
import sys # 导入sys，用于系统相关操作
import networkx as nx # 导入networkx库，用于图形分析和可视化
import matplotlib.pyplot as plt # 导入matplotlib.pyplot，用于绘图
import io # 导入io库，用于内存中的字节流操作
import base64 # 导入base64库，用于图像编码
import matplotlib.font_manager as fm # 导入字体管理器

# 将项目根目录添加到Python路径，以便正确导入模块
# Add project root directory to Python path for correct module imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# 从项目中导入因果推理引擎
# Import the CausalReasoningEngine module from the project
from modules.causal_reasoning import CausalReasoningEngine # 从因果推理模块导入CausalReasoningEngine类

class EnhancedCausalGraphTool(BaseTool):
    """
    增强型因果图构建与分析工具
    Enhanced Causal Graph Building and Analysis Tool

    构建事件影响的详细因果图，分析传导路径，并提供可视化。
    Builds a detailed causal graph of event impacts, analyzes transmission paths, and provides visualization.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True) # 允许任意类型作为字段

    name: str = "增强型因果图分析工具" # 工具的名称
    description: str = ( # 工具的描述
        "构建事件影响的详细因果图，分析传导路径，并返回包含分析结果和可视化图像的JSON字符串。"
        "输入为事件信息的JSON字符串和产业链数据的JSON字符串。"
    )
    causal_reasoning_engine: CausalReasoningEngine # 将causal_reasoning_engine声明为字段

    def __init__(self, causal_reasoning_engine: CausalReasoningEngine, **kwargs):
        """
        初始化增强型因果图分析工具。
        Initialize the Enhanced Causal Graph Analysis Tool.

        Args:
            causal_reasoning_engine (CausalReasoningEngine): 因果推理引擎实例。
                                                              An instance of the CausalReasoningEngine.
        """
        super().__init__(causal_reasoning_engine=causal_reasoning_engine, **kwargs) # 将其传递给Pydantic

    def _run(self, event_info_json: str, industry_chain_json: str) -> str:
        """
        执行增强型因果图构建和分析。
        Execute enhanced causal graph building and analysis.

        Args:
            event_info_json (str): 事件信息的JSON字符串。
                                   A JSON string of the event information.
            industry_chain_json (str): 产业链数据的JSON字符串。
                                       A JSON string of the industry chain data.

        Returns:
            str: 包含分析结果和可视化图像的JSON字符串。
                 A JSON string containing analysis results and visualization image.
        """
        try:
            # 将输入的JSON字符串转换为Python字典
            # Convert the input JSON strings to Python dictionaries
            event_info = json.loads(event_info_json)
            industry_chain = json.loads(industry_chain_json)
            
            # 调用因果推理引擎的create_causal_graph方法构建因果图
            # Call the create_causal_graph method of the causal reasoning engine to build the causal graph
            causal_graph_data = self.causal_reasoning_engine.create_causal_graph(event_info, industry_chain)
            
            # 生成因果图的可视化图像 (base64编码)
            # Generate a visualization image of the causal graph (base64 encoded)
            graph_image_base64 = self._visualize_causal_graph(causal_graph_data)
            
            # 提取关键传导路径
            # Extract key transmission paths
            key_paths = self._extract_key_paths(causal_graph_data)
            
            # 评估置信度
            # Assess confidence
            confidence_assessment = self._assess_confidence(causal_graph_data)
            
            # 构建并返回包含所有信息的JSON字符串
            # Build and return a JSON string containing all information
            result = {
                "causal_graph_analysis": causal_graph_data, # 因果图分析数据
                "visualization_image_base64": graph_image_base64, # 可视化图像 (base64)
                "key_transmission_paths": key_paths, # 关键传导路径
                "confidence_assessment": confidence_assessment # 置信度评估
            }
            return json.dumps(result, ensure_ascii=False) # 返回JSON字符串

        except Exception as e:
            # 如果发生错误，记录错误并返回错误信息的JSON字符串
            # If an error occurs, log the error and return a JSON string with error information
            # self.logger.error(f"增强型因果图构建失败: {str(e)}") # 假设有日志记录器
            return json.dumps({"error": f"增强型因果图构建失败: {str(e)}"}, ensure_ascii=False)

    def _visualize_causal_graph(self, causal_graph_data: dict) -> str:
        """
        将因果图数据可视化为网络图，并返回base64编码的图像字符串。
        Visualize causal graph data as a network graph and return a base64 encoded image string.

        Args:
            causal_graph_data (dict): 因果图数据。
                                      Causal graph data.

        Returns:
            str: base64编码的PNG图像字符串，如果可视化失败则返回错误信息。
                 A base64 encoded PNG image string, or an error message if visualization fails.
        """
        try:
            # 尝试设置中文字体，以正确显示图中的中文标签
            try:
                # 常见的支持中文的字体列表
                common_chinese_fonts_to_try = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
                font_found_and_set = False
                
                # 遍历尝试设置字体
                for font_name in common_chinese_fonts_to_try:
                    try:
                        # 检查字体是否存在。findfont会抛错如果找不到。
                        fm.findfont(fm.FontProperties(family=font_name), fallback_to_default=False)
                        # 如果上面一行没有抛出异常，说明字体系统认为是可用的
                        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif'] # 将找到的字体加到列表前面
                        # print(f"调试：尝试将 {font_name} 设置为中文字体。") # 调试信息
                        font_found_and_set = True
                        break # 找到一个就用
                    except Exception:
                        # print(f"调试：字体 {font_name} 未找到或设置失败。") # 调试信息
                        continue
                
                if not font_found_and_set:
                    # 如果所有尝试的字体都未成功设置，打印警告
                    # 保留 'sans-serif' 作为最后的希望，如果系统默认的 sans-serif 支持中文
                    if 'sans-serif' not in plt.rcParams['font.sans-serif']:
                         plt.rcParams['font.sans-serif'].append('sans-serif')
                    print(
                        "警告: 未能自动找到并设置常见中文字体 (如 SimHei, Microsoft YaHei, WenQuanYi Zen Hei, Arial Unicode MS)。"
                        "图形中的中文可能无法正确显示。请确保您的系统中安装了支持中文的字体，并且 Matplotlib 可以找到它们。"
                    )
                
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
            except Exception as font_setting_error:
                # self.logger.warning(f"设置中文字体参数时发生一般性错误: {font_setting_error}，中文可能无法正确显示。") # 假设有logger
                print(f"警告: 设置中文字体参数时发生一般性错误: {font_setting_error}。中文可能无法正确显示。")
                # 即使字体设置失败，也继续尝试绘图，依赖Matplotlib的默认行为
            
            G = nx.DiGraph() # 创建一个有向图
            
            # 添加节点和边 (简化示例，实际应根据causal_graph_data的结构添加)
            # Add nodes and edges (simplified example, should be based on the structure of causal_graph_data)
            
            # 示例：添加事件节点
            # Example: Add event node
            event_desc = causal_graph_data.get("event_info", {}).get("basic_info", {}).get("event_description", "事件")
            G.add_node(event_desc, type='event', color='skyblue')

            # 示例：添加直接影响的环节节点和边
            # Example: Add directly affected segment nodes and edges
            direct_impacts = causal_graph_data.get("direct_causality", {}).get("directly_affected_segments", [])
            for i, segment_info in enumerate(direct_impacts):
                segment_name = segment_info.get("segment_name", f"环节{i+1}")
                G.add_node(segment_name, type='segment', color='lightgreen')
                G.add_edge(event_desc, segment_name, label=segment_info.get("impact_mechanism", "直接影响"))

            # 示例：添加传导路径中的节点和边
            # Example: Add nodes and edges from transmission paths
            paths = causal_graph_data.get("transmission_paths", [])
            for path in paths:
                steps = path.get("steps", [])
                for step in steps:
                    from_node = step.get("from_element", "未知源")
                    to_node = step.get("to_element", "未知目标")
                    mechanism = step.get("mechanism", "传导")
                    if not G.has_node(from_node):
                        G.add_node(from_node, type='intermediate', color='lightcoral')
                    if not G.has_node(to_node):
                        G.add_node(to_node, type='intermediate', color='lightcoral')
                    if not G.has_edge(from_node, to_node):
                         G.add_edge(from_node, to_node, label=mechanism[:15]) # 标签长度限制

            if not G.nodes(): # 如果图中没有节点，则无法绘图
                return "无法生成图像：图中无节点。"

            plt.figure(figsize=(12, 8)) # 设置图像大小
            pos = nx.spring_layout(G, k=0.5, iterations=50) # 使用spring布局算法排列节点
            
            # 获取节点颜色
            # Get node colors
            node_colors = [data.get('color', 'lightgrey') for node, data in G.nodes(data=True)]
            
            nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_colors, font_size=8, font_weight='bold', arrows=True, arrowstyle='->', arrowsize=20) # 绘制图形
            edge_labels = nx.get_edge_attributes(G, 'label') # 获取边的标签
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7) # 绘制边的标签
            
            plt.title("因果关系图", fontsize=15) # 设置图像标题
            
            # 将图像保存到内存中的字节流
            # Save the image to an in-memory bytes stream
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            plt.close() # 关闭matplotlib图形，释放资源
            
            # 将图像字节流编码为base64字符串
            # Encode the image bytes stream to a base64 string
            img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            return f"data:image/png;base64,{img_base64}" # 返回base64编码的图像数据

        except Exception as vis_e:
            # self.logger.error(f"因果图可视化失败: {str(vis_e)}") # 假设有日志记录器
            return f"图像生成失败: {str(vis_e)}"

    def _extract_key_paths(self, causal_graph_data: dict) -> list:
        """
        从因果图数据中提取关键的传导路径。
        Extract key transmission paths from causal graph data.

        Args:
            causal_graph_data (dict): 因果图数据。
                                      Causal graph data.

        Returns:
            list: 关键传导路径列表，按置信度或重要性排序。
                  A list of key transmission paths, sorted by confidence or importance.
        """
        paths = causal_graph_data.get("transmission_paths", [])
        # 根据置信度和路径长度等因素对路径进行排序和筛选
        # Sort and filter paths based on factors like confidence and path length
        # 此处为简化实现，直接返回前3条路径或所有路径（如果少于3条）
        # Simplified implementation: return the top 3 paths or all paths if fewer than 3
        sorted_paths = sorted(paths, key=lambda p: p.get("confidence", 0.0) * (1.0 / p.get("path_length", 1.0)), reverse=True)
        return sorted_paths[:3]

    def _assess_confidence(self, causal_graph_data: dict) -> dict:
        """
        评估整个因果分析的置信度。
        Assess the confidence of the entire causal analysis.

        Args:
            causal_graph_data (dict): 因果图数据。
                                      Causal graph data.

        Returns:
            dict: 包含总体置信度和详细说明的字典。
                  A dictionary containing overall confidence and detailed explanation.
        """
        # 基于因果图中的各种置信度指标计算综合置信度
        # Calculate composite confidence based on various confidence indicators in the causal graph
        overall_strength = causal_graph_data.get("causal_strength", {}).get("overall_strength", 0.0)
        uncertainty = causal_graph_data.get("uncertainty_analysis", {}).get("overall_uncertainty", 1.0)
        
        # 综合评估，可以根据具体需求调整权重
        # Comprehensive assessment, weights can be adjusted based on specific needs
        confidence_score = (overall_strength * 0.6) + ((1 - uncertainty) * 0.4)
        
        description = "置信度基于因果强度和不确定性分析综合评估。"
        if confidence_score > 0.75:
            level = "高"
        elif confidence_score > 0.5:
            level = "中"
        else:
            level = "低"
            
        return {
            "overall_confidence_score": round(confidence_score, 2), # 总体置信度分数
            "confidence_level": level, # 置信度等级
            "assessment_description": description # 评估描述
        }

# 此处可以继续添加其他与因果分析相关的增强工具类
# Other enhanced tool classes related to causal analysis can be added here
