"""
可视化工具模块
Visualization Utilities Module

提供生成各种图表的功能，如图网络图、流程图、树状图等
Provides functions to generate various charts like network graphs, flowcharts, tree diagrams, etc.
"""

import base64 # 导入base64模块，用于图像编码
from io import BytesIO # 导入BytesIO，用于内存中的二进制流处理
from typing import List, Dict, Any, Optional # 导入类型提示相关的模块

# 导入第三方库 - Import third-party libraries
try:
    import matplotlib # 尝试导入matplotlib库
    import matplotlib.pyplot as plt # 导入matplotlib的pyplot模块，用于绘图
    import networkx as nx # 导入networkx库，用于网络图分析和绘制
    import plotly.graph_objects as go # 导入plotly的graph_objects模块，用于交互式图表
    import plotly.express as px # 导入plotly的express模块，用于快速绘图
    HAS_VISUALIZATION_LIBS = True # 设置标志位，表示可视化库可用
except ImportError:
    HAS_VISUALIZATION_LIBS = False # 设置标志位，表示可视化库不可用
    # 记录一个警告或错误，如果这些库是必需的
    # self.logger.warning("Matplotlib, NetworkX, or Plotly not installed. Visualization capabilities will be limited.")

# 导入自定义数据结构 - Import custom data structures
from .visualization_data import GraphData, TreeNode, FlowchartData # 从同级目录的visualization_data模块导入数据结构类

class VisualizationGenerator:
    """
    可视化图表生成器类
    Visualization Chart Generator Class
    """

    def __init__(self, logger=None): # 构造函数，可选接收一个logger实例
        """
        初始化可视化生成器
        Initialize Visualization Generator
        """
        self.logger = logger # 保存logger实例
        if not HAS_VISUALIZATION_LIBS: # 检查可视化库是否可用
            if self.logger: # 如果有logger实例
                self.logger.warning( # 记录警告信息
                    "Matplotlib, NetworkX, or Plotly not installed. "
                    "Visualization capabilities will be limited."
                )
            else: # 如果没有logger实例
                print( # 打印警告信息
                    "Warning: Matplotlib, NetworkX, or Plotly not installed. "
                    "Visualization capabilities will be limited."
                )

    def _check_libs(self) -> bool: # 定义检查库是否可用的内部方法
        """检查核心可视化库是否已安装 - Check if core visualization libraries are installed"""
        if not HAS_VISUALIZATION_LIBS: # 如果库不可用
            if self.logger: self.logger.error("Required visualization libraries are not installed.") # 记录错误日志
            return False # 返回False
        return True # 返回True

    def _to_base64_image(self, fig: Any, fig_format: str = 'png') -> Optional[str]: # 定义将图表转换为Base64编码图像的内部方法
        """
        将Matplotlib或Plotly图表对象转换为Base64编码的图像字符串
        Convert Matplotlib or Plotly figure object to a Base64 encoded image string

        Args:
            fig: Matplotlib Figure or Plotly Figure object
            fig_format: 图像格式 ('png', 'jpeg', 'svg') - Image format

        Returns:
            Optional[str]: Base64编码的图像字符串，如果失败则返回None
                           Base64 encoded image string, or None if failed
        """
        if not self._check_libs(): return None # 检查库是否可用，不可用则返回None

        try:
            buffer = BytesIO() # 创建内存中的二进制流对象
            if isinstance(fig, matplotlib.figure.Figure): # 如果是Matplotlib图表对象
                fig.savefig(buffer, format=fig_format, bbox_inches='tight') # 保存图表到buffer
                plt.close(fig) # 关闭图表以释放内存
            elif isinstance(fig, go.Figure): # 如果是Plotly图表对象
                if fig_format == 'svg': # 如果格式是svg
                    fig.write_image(buffer, format=fig_format) # 直接写入svg格式
                else: # 其他格式
                    fig.write_image(buffer, format=fig_format, scale=2) # 写入其他格式，放大2倍以提高清晰度
            else: # 如果图表类型不支持
                if self.logger: self.logger.error(f"Unsupported figure type: {type(fig)}") # 记录错误日志
                return None # 返回None

            buffer.seek(0) # 将buffer的指针移到开头
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8') # 获取buffer内容并进行Base64编码
            return image_base64 # 返回Base64编码的图像字符串
        except Exception as e: # 捕获异常
            if self.logger: self.logger.error(f"Error converting figure to base64: {e}", exc_info=True) # 记录错误日志
            return None # 返回None

    def create_transmission_path_chart(self, graph_data: GraphData, title: str = "事件传导路径图") -> Optional[str]:
        """
        创建事件传导路径图 (使用NetworkX和Matplotlib)
        Create event transmission path chart (using NetworkX and Matplotlib)

        Args:
            graph_data: GraphData实例，包含节点和边 - GraphData instance with nodes and edges
            title: 图表标题 - Chart title

        Returns:
            Optional[str]: Base64编码的PNG图像字符串，或None
                           Base64 encoded PNG image string, or None
        """
        if not self._check_libs(): return None # 检查库

        try:
            G = nx.DiGraph() # 创建有向图实例
            # 添加节点和边 - Add nodes and edges
            for node in graph_data.nodes:
                G.add_node(node.id, label=node.label, type=node.type, **(node.properties or {}))
            for edge in graph_data.edges:
                G.add_edge(edge.source, edge.target, label=edge.label, weight=edge.weight or 1.0, **(edge.properties or {}))

            plt.style.use('seaborn-v0_8-whitegrid') # 使用seaborn样式
            fig, ax = plt.subplots(figsize=(16, 12)) # 创建图表和子图，设置大小
            
            # 使用更适合有向图的布局 - Use a layout more suitable for directed graphs
            try:
                # 尝试使用graphviz布局（如果安装了pygraphviz）- Try graphviz layout if pygraphviz is installed
                pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
            except ImportError:
                # 回退到spring_layout - Fallback to spring_layout
                if self.logger: self.logger.info("pygraphviz not found, using spring_layout for transmission path chart.")
                pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42) # 弹簧布局

            node_labels = nx.get_node_attributes(G, 'label') # 获取节点标签
            edge_labels = nx.get_edge_attributes(G, 'label') # 获取边标签

            # 根据节点类型设置颜色 - Set node colors based on type
            node_colors = []
            color_map = {"event": "skyblue", "industry": "lightgreen", "company": "salmon", "segment": "gold"}
            for node_id in G.nodes():
                node_type = G.nodes[node_id].get('type', 'default')
                node_colors.append(color_map.get(node_type, 'lightgrey'))

            nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.9, ax=ax) # 绘制节点
            nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color="gray", arrows=True, arrowstyle="-|>", arrowsize=20, ax=ax, connectionstyle="arc3,rad=0.1") # 绘制边
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, ax=ax) # 绘制节点标签
            if edge_labels: # 如果有边标签
                 nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax) # 绘制边标签

            ax.set_title(title, fontsize=18) # 设置图表标题
            plt.axis('off') # 关闭坐标轴
            plt.tight_layout() # 调整布局

            return self._to_base64_image(fig) # 返回Base64编码的图像
        except Exception as e: # 捕获异常
            if self.logger: self.logger.error(f"Error creating transmission path chart: {e}", exc_info=True) # 记录错误日志
            return None # 返回None

    def create_industry_chain_network(self, graph_data: GraphData, title: str = "产业链网络图") -> Optional[str]:
        """
        创建产业链网络图 (使用Plotly，支持交互)
        Create industry chain network graph (using Plotly, interactive)

        Args:
            graph_data: GraphData实例 - GraphData instance
            title: 图表标题 - Chart title

        Returns:
            Optional[str]: Base64编码的PNG图像字符串，或None
                           Base64 encoded PNG image string, or None
        """
        if not self._check_libs(): return None # 检查库

        try:
            G = nx.Graph() # 创建无向图实例
            node_map = {node.id: i for i, node in enumerate(graph_data.nodes)} # 创建节点ID到索引的映射
            
            # 添加节点和边 - Add nodes and edges
            for node in graph_data.nodes:
                G.add_node(node.id, label=node.label, type=node.type, **(node.properties or {}))
            for edge in graph_data.edges:
                G.add_edge(edge.source, edge.target, label=edge.label, weight=edge.weight or 1.0)

            pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42) # 弹簧布局

            edge_x, edge_y = [], [] # 初始化边坐标列表
            for edge_obj in G.edges():
                x0, y0 = pos[edge_obj[0]]
                x1, y1 = pos[edge_obj[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            node_x, node_y, node_text, node_color, node_size = [], [], [], [], [] # 初始化节点属性列表
            color_map = {"upstream": "blue", "midstream": "green", "downstream": "red", "company":"purple", "segment":"orange"}
            for node_id in G.nodes():
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)
                node_info = G.nodes[node_id]
                node_text.append(f"{node_info.get('label', node_id)}<br>Type: {node_info.get('type', 'N/A')}")
                node_color.append(color_map.get(node_info.get('type', 'default'), 'grey'))
                # 根据度数设置节点大小 - Set node size based on degree
                node_size.append(15 + G.degree(node_id) * 5)


            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines') # 创建边的散点图轨迹
            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=[G.nodes[nid]['label'] for nid in G.nodes()], textposition="bottom center",
                                    hovertext=node_text, hoverinfo='text',
                                    marker=dict(showscale=False, color=node_color, size=node_size, line_width=2)) # 创建节点的散点图轨迹

            fig = go.Figure(data=[edge_trace, node_trace], # 创建Plotly图表对象
                            layout=go.Layout(
                                title=title, titlefont_size=16, showlegend=False, hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                            )
            return self._to_base64_image(fig) # 返回Base64编码的图像
        except Exception as e: # 捕获异常
            if self.logger: self.logger.error(f"Error creating industry chain network: {e}", exc_info=True) # 记录错误日志
            return None # 返回None

    def create_stock_pool_hierarchy(self, tree_data: TreeNode, title: str = "股票池层次图") -> Optional[str]:
        """
        创建股票池层次图 (使用Plotly Treemap)
        Create stock pool hierarchy chart (using Plotly Treemap)

        Args:
            tree_data: TreeNode根节点实例 - TreeNode root node instance
            title: 图表标题 - Chart title

        Returns:
            Optional[str]: Base64编码的PNG图像字符串，或None
                           Base64 encoded PNG image string, or None
        """
        if not self._check_libs(): return None # 检查库

        try:
            ids, labels, parents, values = [], [], [], [] # 初始化列表

            def _traverse_tree(node: TreeNode, parent_id: Optional[str] = None): # 定义遍历树的内部函数
                ids.append(node.id) # 添加节点ID
                labels.append(node.name) # 添加节点名称
                parents.append(parent_id if parent_id else "") # 添加父节点ID，根节点父ID为空字符串
                values.append(node.value if node.value is not None else 1) # 添加节点值，如果为空则为1

                for child in node.children: # 遍历子节点
                    _traverse_tree(child, node.id) # 递归调用

            _traverse_tree(tree_data) # 从根节点开始遍历

            fig = go.Figure(go.Treemap( # 创建Treemap图表对象
                ids=ids,
                labels=labels,
                parents=parents,
                values=values,
                root_color="lightgrey",
                textinfo="label+value+percent parent" # 显示文本信息
            ))
            fig.update_layout(title_text=title, margin=dict(t=50, l=25, r=25, b=25)) # 更新布局，设置标题和边距
            return self._to_base64_image(fig) # 返回Base64编码的图像
        except Exception as e: # 捕获异常
            if self.logger: self.logger.error(f"Error creating stock pool hierarchy: {e}", exc_info=True) # 记录错误日志
            return None # 返回None

    def create_flowchart(self, flowchart_data: FlowchartData, title: str = "分析流程图") -> Optional[str]:
        """
        创建分析流程图 (使用Plotly Sankey Diagram 或 NetworkX)
        Create analysis flowchart (using Plotly Sankey Diagram or NetworkX)
        
        此实现使用NetworkX和Matplotlib创建一个简单的流程图。
        This implementation uses NetworkX and Matplotlib to create a simple flowchart.

        Args:
            flowchart_data: FlowchartData实例 - FlowchartData instance
            title: 图表标题 - Chart title

        Returns:
            Optional[str]: Base64编码的PNG图像字符串，或None
                           Base64 encoded PNG image string, or None
        """
        if not self._check_libs(): return None # 检查库

        try:
            G = nx.DiGraph() # 创建有向图实例
            node_labels = {} # 初始化节点标签字典
            for step in flowchart_data.steps: # 遍历步骤
                G.add_node(step.id, label=step.label, type=step.type) # 添加节点
                node_labels[step.id] = step.label # 设置节点标签
            
            for conn in flowchart_data.connections: # 遍历连接
                G.add_edge(conn['from'], conn['to']) # 添加边

            plt.style.use('seaborn-v0_8-whitegrid') # 使用seaborn样式
            fig, ax = plt.subplots(figsize=(12, 8)) # 创建图表和子图

            try:
                pos = nx.nx_agraph.graphviz_layout(G, prog="dot") # 尝试使用graphviz布局
            except ImportError:
                if self.logger: self.logger.info("pygraphviz not found, using spring_layout for flowchart.") # 记录信息
                pos = nx.spring_layout(G, k=0.8, iterations=30, seed=42) # 回退到spring布局

            # 定义节点形状和颜色 - Define node shapes and colors
            node_shapes = {'start': 'o', 'end': 'o', 'process': 's', 'decision': 'd', 'io': 'p'}
            node_colors_map = {'start': 'lightgreen', 'end': 'salmon', 'process': 'skyblue', 'decision': 'gold', 'io': 'lightcoral'}
            
            for node_type, shape in node_shapes.items(): # 遍历节点类型和形状
                node_list = [n for n, attr in G.nodes(data=True) if attr.get('type') == node_type] # 获取对应类型的节点列表
                color = node_colors_map.get(node_type, 'lightgrey') # 获取颜色
                nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_shape=shape, node_size=3500, node_color=color, alpha=0.9, ax=ax) # 绘制节点

            nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowstyle="-|>", arrowsize=15, ax=ax, connectionstyle="arc3,rad=0.1") # 绘制边
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, ax=ax) # 绘制节点标签

            ax.set_title(title, fontsize=16) # 设置标题
            plt.axis('off') # 关闭坐标轴
            plt.tight_layout() # 调整布局

            return self._to_base64_image(fig) # 返回Base64编码的图像
        except Exception as e: # 捕获异常
            if self.logger: self.logger.error(f"Error creating flowchart: {e}", exc_info=True) # 记录错误日志
            return None # 返回None

if __name__ == '__main__':
    # 示例用法 - Example Usage (需要安装matplotlib, networkx, plotly)
    # from utils.logger import setup_logger # 假设有一个setup_logger函数
    # test_logger = setup_logger() # 创建测试logger
    test_logger = None # 或者不使用logger进行简单测试 - Or test simply without logger

    generator = VisualizationGenerator(logger=test_logger) # 创建生成器实例

    # 1. 测试传导路径图 - Test Transmission Path Chart
    nodes_trans_data = [
        {"id": "event", "label": "新闻:补贴调整", "type": "event", "properties": {"date": "2023-01-01"}},
        {"id": "industryA", "label": "行业:新能源车", "type": "industry"},
        {"id": "segment1", "label": "环节:电池", "type": "segment"},
        {"id": "companyX", "label": "公司:CATL", "type": "company"}
    ]
    edges_trans_data = [
        {"source": "event", "target": "industryA", "label": "直接影响", "weight": 0.9},
        {"source": "industryA", "target": "segment1", "label": "需求传导", "weight": 0.8},
        {"source": "segment1", "target": "companyX", "label": "市场份额", "weight": 0.7}
    ]
    trans_graph = GraphData(nodes=nodes_trans_data, edges=edges_trans_data) # type: ignore
    trans_img = generator.create_transmission_path_chart(trans_graph, title="新能源补贴政策传导图")
    if trans_img: print("传导路径图已生成 (Base64)") # 打印信息
    # with open("transmission_path.png", "wb") as f: f.write(base64.b64decode(trans_img)) # 可选：保存到文件

    # 2. 测试产业链网络图 - Test Industry Chain Network
    nodes_chain_data = [
        {"id": "lithium", "label": "锂矿", "type": "upstream"},
        {"id": "battery_material", "label": "电池材料", "type": "upstream"},
        {"id": "battery", "label": "动力电池", "type": "midstream"},
        {"id": "ev_mfg", "label": "整车制造", "type": "midstream"},
        {"id": "sales", "label": "销售服务", "type": "downstream"}
    ]
    edges_chain_data = [
        {"source": "lithium", "target": "battery_material", "label": "供应"},
        {"source": "battery_material", "target": "battery", "label": "供应"},
        {"source": "battery", "target": "ev_mfg", "label": "供应"},
        {"source": "ev_mfg", "target": "sales", "label": "销售"}
    ]
    chain_graph = GraphData(nodes=nodes_chain_data, edges=edges_chain_data) # type: ignore
    chain_img = generator.create_industry_chain_network(chain_graph, title="新能源汽车产业链")
    if chain_img: print("产业链网络图已生成 (Base64)") # 打印信息

    # 3. 测试股票池层次图 - Test Stock Pool Hierarchy
    stock_tree_data_raw = {
        "id": "ev_pool", "name": "新能源车股票池", "value": 100,
        "children": [
            {"id": "upstream_pool", "name": "上游材料", "value": 30, "children": [
                {"id": "ganfeng", "name": "赣锋锂业", "value": 15, "properties": {"code": "002460"}},
                {"id": "tianqi", "name": "天齐锂业", "value": 15, "properties": {"code": "002466"}}
            ]},
            {"id": "midstream_pool", "name": "中游制造", "value": 70, "children": [
                {"id": "byd", "name": "比亚迪", "value": 40, "properties": {"code": "002594"}},
                {"id": "catl", "name": "宁德时代", "value": 30, "properties": {"code": "300750"}}
            ]}
        ]
    }
    # 需要将原始字典转换为TreeNode对象 - Need to convert raw dict to TreeNode object
    # (假设 VisualizationDataFactory.create_stock_pool_tree_data 已正确处理)
    # 为了直接测试，我们先手动构建TreeNode - For direct testing, manually build TreeNode
    
    # 手动构建 TreeNode 的辅助函数 - Helper function to manually build TreeNode
    def dict_to_treenode(d: Dict) -> TreeNode:
        children = [dict_to_treenode(c) for c in d.get("children", [])]
        return TreeNode(id=d["id"], name=d["name"], value=d.get("value"), children=children, properties=d.get("properties", {}))

    stock_tree_obj = dict_to_treenode(stock_tree_data_raw)
    stock_hierarchy_img = generator.create_stock_pool_hierarchy(stock_tree_obj, title="新能源股票池分层")
    if stock_hierarchy_img: print("股票池层次图已生成 (Base64)") # 打印信息

    # 4. 测试流程图 - Test Flowchart
    flow_steps_data = [
        {"id": "s1", "label": "新闻输入", "type": "io"},
        {"id": "s2", "label": "行业分析", "type": "process"},
        {"id": "s3", "label": "产业链分析", "type": "process"},
        {"id": "s4", "label": "股票池构建", "type": "process"},
        {"id": "s5", "label": "投资决策", "type": "process"},
        {"id": "s6", "label": "输出报告", "type": "io"}
    ]
    flow_connections_data = [
        {"from": "s1", "to": "s2"}, {"from": "s2", "to": "s3"},
        {"from": "s3", "to": "s4"}, {"from": "s4", "to": "s5"},
        {"from": "s5", "to": "s6"}
    ]
    flow_chart_obj = FlowchartData(steps=flow_steps_data, connections=flow_connections_data) # type: ignore
    flowchart_img = generator.create_flowchart(flow_chart_obj, title="投资分析流程")
    if flowchart_img: print("流程图已生成 (Base64)") # 打印信息
