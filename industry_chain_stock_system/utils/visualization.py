"""
可视化工具模块
Visualization Utility Module

提供数据可视化功能，如图表生成等
Provides data visualization functionality, such as chart generation, etc.
"""

import json
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# 导入日志工具 - Import logging utility
from .logger import LoggerMixin

# 设置中文字体 - Setup Chinese font
# 注意：需要确保系统中有可用的中文字体，例如 "SimHei" 或 "Microsoft YaHei"
# Note: Ensure that a Chinese font is available in the system, e.g., "SimHei" or "Microsoft YaHei"
try:
    # 尝试查找常用中文字体 - Try to find common Chinese fonts
    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    
    # 优先选择的字体列表 - Preferred font list
    preferred_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
    
    default_font = None
    for font_name in preferred_fonts:
        for font_path in font_paths:
            if font_name.lower() in font_path.lower():
                default_font = fm.FontProperties(fname=font_path).get_name()
                break
        if default_font:
            break
    
    if default_font:
        plt.rcParams['font.sans-serif'] = [default_font]
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题 - Solve minus sign display issue
    else:
        print("警告：未找到合适的中文字体，图表中文可能无法正常显示。")
        print("Warning: No suitable Chinese font found, Chinese characters in charts may not display correctly.")

except Exception as e:
    print(f"设置中文字体失败: {e}")
    print("Warning: Failed to set Chinese font, Chinese characters in charts may not display correctly.")

class VisualizationUtils(LoggerMixin):
    """
    可视化工具类
    Visualization Utility Class
    """
    
    def __init__(self):
        """初始化可视化工具 - Initialize visualization utility"""
        self.logger.info("可视化工具初始化完成")
        
        # 默认颜色方案 - Default color scheme
        self.colors = {
            "primary": "#007bff",
            "secondary": "#6c757d",
            "success": "#28a745",
            "danger": "#dc3545",
            "warning": "#ffc107",
            "info": "#17a2b8",
            "light": "#f8f9fa",
            "dark": "#343a40"
        }
    
    def plot_bar_chart(
        self, 
        data: Dict[str, float], 
        title: str, 
        xlabel: str, 
        ylabel: str,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        绘制条形图
        Plot bar chart
        
        Args:
            data: 数据字典 {标签: 值} - Data dictionary {label: value}
            title: 图表标题 - Chart title
            xlabel: X轴标签 - X-axis label
            ylabel: Y轴标签 - Y-axis label
            filename: 保存文件名 (可选) - Save filename (optional)
            
        Returns:
            Optional[str]: 文件路径或None - File path or None
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(data.keys(), data.values(), color=self.colors["primary"])
            plt.title(title, fontsize=16)
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            if filename:
                plt.savefig(filename)
                self.logger.info(f"条形图已保存到: {filename}")
                plt.close()
                return filename
            else:
                plt.show()
                plt.close()
                return None
                
        except Exception as e:
            self.logger.error(f"绘制条形图失败: {e}")
            return None
            
    def plot_pie_chart(
        self, 
        data: Dict[str, float], 
        title: str,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        绘制饼图
        Plot pie chart
        
        Args:
            data: 数据字典 {标签: 值} - Data dictionary {label: value}
            title: 图表标题 - Chart title
            filename: 保存文件名 (可选) - Save filename (optional)
            
        Returns:
            Optional[str]: 文件路径或None - File path or None
        """
        try:
            plt.figure(figsize=(8, 8))
            plt.pie(
                data.values(), 
                labels=data.keys(), 
                autopct='%1.1f%%', 
                startangle=90,
                colors=[self.colors[c] for c in ["primary", "success", "warning", "info", "danger"][:len(data)]]
            )
            plt.title(title, fontsize=16)
            plt.axis('equal') # 确保饼图是圆的 - Ensure pie chart is circular
            plt.tight_layout()
            
            if filename:
                plt.savefig(filename)
                self.logger.info(f"饼图已保存到: {filename}")
                plt.close()
                return filename
            else:
                plt.show()
                plt.close()
                return None
                
        except Exception as e:
            self.logger.error(f"绘制饼图失败: {e}")
            return None
            
    def plot_network_graph(
        self, 
        nodes: List[str], 
        edges: List[tuple[str, str, float]], 
        title: str,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        绘制网络图
        Plot network graph
        
        Args:
            nodes: 节点列表 - List of nodes
            edges: 边列表 [(源, 目标, 权重)] - List of edges [(source, target, weight)]
            title: 图表标题 - Chart title
            filename: 保存文件名 (可选) - Save filename (optional)
            
        Returns:
            Optional[str]: 文件路径或None - File path or None
        """
        try:
            G = nx.Graph()
            G.add_nodes_from(nodes)
            
            # 添加带权重的边 - Add weighted edges
            weighted_edges = [(u, v, {"weight": w}) for u, v, w in edges]
            G.add_edges_from(weighted_edges)
            
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, k=0.5, iterations=50) # 布局算法 - Layout algorithm
            
            # 节点大小根据度数调整 - Node size adjusted by degree
            node_sizes = [G.degree(node) * 100 + 200 for node in G.nodes()]
            
            # 边宽度根据权重调整 - Edge width adjusted by weight
            edge_weights = [d["weight"] * 5 for u, v, d in G.edges(data=True)]
            
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=self.colors["info"], alpha=0.8)
            nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=self.colors["secondary"], alpha=0.6)
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            plt.title(title, fontsize=16)
            plt.axis('off') # 关闭坐标轴 - Turn off axis
            plt.tight_layout()
            
            if filename:
                plt.savefig(filename)
                self.logger.info(f"网络图已保存到: {filename}")
                plt.close()
                return filename
            else:
                plt.show()
                plt.close()
                return None
                
        except Exception as e:
            self.logger.error(f"绘制网络图失败: {e}")
            return None
            
    def plot_sankey_diagram(
        self, 
        data: Dict[str, Any], 
        title: str,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        绘制桑基图 (使用Plotly)
        Plot Sankey diagram (using Plotly)
        
        Args:
            data: 桑基图数据，格式：
                  {
                      "labels": ["节点1", "节点2", ...],
                      "source": [源节点索引],
                      "target": [目标节点索引],
                      "value": [流量值]
                  }
                  Sankey diagram data, format:
                  {
                      "labels": ["Node1", "Node2", ...],
                      "source": [source node indices],
                      "target": [target node indices],
                      "value": [flow values]
                  }
            title: 图表标题 - Chart title
            filename: 保存文件名 (可选, HTML格式) - Save filename (optional, HTML format)
            
        Returns:
            Optional[str]: 文件路径或None - File path or None
        """
        try:
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=data["labels"],
                    color=[self.colors[c] for c in ["primary", "success", "info", "warning"] * (len(data["labels"]) // 4 + 1)]
                ),
                link=dict(
                    source=data["source"],
                    target=data["target"],
                    value=data["value"]
                ))])
            
            fig.update_layout(title_text=title, font_size=10)
            
            if filename:
                fig.write_html(filename)
                self.logger.info(f"桑基图已保存到: {filename}")
                return filename
            else:
                fig.show()
                return None
                
        except Exception as e:
            self.logger.error(f"绘制桑基图失败: {e}")
            return None

# 便捷函数 - Convenience functions
_viz_utils_instance = None

def get_visualization_utils() -> VisualizationUtils:
    """获取VisualizationUtils单例 - Get VisualizationUtils singleton instance"""
    global _viz_utils_instance
    if _viz_utils_instance is None:
        _viz_utils_instance = VisualizationUtils()
    return _viz_utils_instance

if __name__ == "__main__":
    # 测试可视化工具 - Test visualization utility
    viz = get_visualization_utils()
    
    # 测试条形图 - Test bar chart
    bar_data = {"A": 10, "B": 20, "C": 15, "D": 25}
    viz.plot_bar_chart(bar_data, "示例条形图", "类别", "数值", "output/bar_chart_test.png")
    
    # 测试饼图 - Test pie chart
    pie_data = {"苹果": 40, "香蕉": 30, "橙子": 20, "葡萄": 10}
    viz.plot_pie_chart(pie_data, "水果占比饼图", "output/pie_chart_test.png")
    
    # 测试网络图 - Test network graph
    nodes = ["A", "B", "C", "D", "E"]
    edges = [("A", "B", 0.8), ("A", "C", 0.5), ("B", "D", 0.9), ("C", "E", 0.7), ("D", "E", 0.6)]
    viz.plot_network_graph(nodes, edges, "示例网络关系图", "output/network_graph_test.png")
    
    # 测试桑基图 - Test Sankey diagram
    sankey_data = {
        "labels": ["事件A", "环节X", "环节Y", "结果1", "结果2"],
        "source": [0, 0, 1, 1, 2, 2],
        "target": [1, 2, 3, 4, 3, 4],
        "value": [8, 2, 3, 5, 1, 1]
    }
    viz.plot_sankey_diagram(sankey_data, "示例桑基图", "output/sankey_diagram_test.html")
    
    print("可视化工具测试完成，请检查output目录下的图表文件")
