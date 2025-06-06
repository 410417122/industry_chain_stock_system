"""
可视化数据结构模块
Visualization Data Structure Module

定义用于生成图表所需的标准化数据结构
Defines standardized data structures required for chart generation
"""

from typing import List, Dict, Any, Optional # 导入类型提示相关的模块
from pydantic import BaseModel, Field # 从pydantic导入BaseModel和Field，用于数据验证和模型定义

# 定义节点模型 - Define Node model
class Node(BaseModel):
    """
    图表中的节点模型
    Node model for charts
    """
    id: str = Field(..., description="节点唯一ID - Unique ID for the node") # 节点ID，必需字段
    label: str = Field(..., description="节点显示标签 - Display label for the node") # 节点标签，必需字段
    type: Optional[str] = Field(None, description="节点类型（可选）- Optional node type") # 节点类型，可选字段
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="节点属性（可选）- Optional node properties") # 节点属性，默认为空字典

# 定义边模型 - Define Edge model
class Edge(BaseModel):
    """
    图表中的边模型
    Edge model for charts
    """
    source: str = Field(..., description="起始节点ID - Source node ID") # 起始节点ID，必需字段
    target: str = Field(..., description="目标节点ID - Target node ID") # 目标节点ID，必需字段
    label: Optional[str] = Field(None, description="边显示标签（可选）- Optional edge label") # 边标签，可选字段
    weight: Optional[float] = Field(None, description="边权重（可选）- Optional edge weight") # 边权重，可选字段
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="边属性（可选）- Optional edge properties") # 边属性，默认为空字典

# 定义图数据模型 - Define Graph Data model
class GraphData(BaseModel):
    """
    图表数据模型，包含节点和边
    Graph data model, containing nodes and edges
    """
    nodes: List[Node] = Field(..., description="节点列表 - List of nodes") # 节点列表，必需字段
    edges: List[Edge] = Field(..., description="边列表 - List of edges") # 边列表，必需字段
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="图元数据（可选）- Optional graph metadata") # 图元数据，默认为空字典

# 定义树节点模型 - Define Tree Node model
class TreeNode(BaseModel):
    """
    树状图中的节点模型
    Node model for tree charts
    """
    id: str = Field(..., description="节点唯一ID - Unique ID for the node") # 节点ID，必需字段
    name: str = Field(..., description="节点名称 - Name of the node") # 节点名称，必需字段
    value: Optional[Any] = Field(None, description="节点值（可选）- Optional node value") # 节点值，可选字段
    children: Optional[List['TreeNode']] = Field(default_factory=list, description="子节点列表（可选）- Optional list of child nodes") # 子节点列表，默认为空列表
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="节点属性（可选）- Optional node properties") # 节点属性，默认为空字典

# 更新TreeNode的前向引用 - Update forward reference for TreeNode
TreeNode.update_forward_refs()

# 定义流程图步骤模型 - Define Flowchart Step model
class FlowchartStep(BaseModel):
    """
    流程图中的步骤模型
    Step model for flowcharts
    """
    id: str = Field(..., description="步骤唯一ID - Unique ID for the step") # 步骤ID，必需字段
    label: str = Field(..., description="步骤显示标签 - Display label for the step") # 步骤标签，必需字段
    type: str = Field(..., description="步骤类型（如'process', 'decision', 'io'）- Step type (e.g., 'process', 'decision', 'io')") # 步骤类型，必需字段
    next_steps: Optional[List[str]] = Field(default_factory=list, description="后续步骤ID列表（可选）- Optional list of next step IDs") # 后续步骤ID列表，默认为空列表
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="步骤属性（可选）- Optional step properties") # 步骤属性，默认为空字典

# 定义流程图数据模型 - Define Flowchart Data model
class FlowchartData(BaseModel):
    """
    流程图数据模型
    Flowchart data model
    """
    steps: List[FlowchartStep] = Field(..., description="步骤列表 - List of steps") # 步骤列表，必需字段
    connections: List[Dict[str, str]] = Field(..., description="连接列表（例如 {'from': 'step1_id', 'to': 'step2_id'}）- List of connections (e.g., {'from': 'step1_id', 'to': 'step2_id'})") # 连接列表，必需字段
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="流程图元数据（可选）- Optional flowchart metadata") # 流程图元数据，默认为空字典

class VisualizationDataFactory:
    """
    可视化数据工厂类
    Factory class for creating visualization data instances
    """

    @staticmethod
    def create_transmission_graph_data(nodes_data: List[Dict], edges_data: List[Dict], metadata: Optional[Dict] = None) -> GraphData:
        """
        创建传导路径图的GraphData实例
        Create GraphData instance for transmission path graph

        Args:
            nodes_data (List[Dict]): 节点数据列表，每个字典包含id, label, type, properties
            edges_data (List[Dict]): 边数据列表，每个字典包含source, target, label, weight, properties
            metadata (Optional[Dict]): 图的元数据

        Returns:
            GraphData: GraphData实例
        """
        # 创建节点对象列表 - Create list of Node objects
        nodes = [Node(**node_data) for node_data in nodes_data]
        # 创建边对象列表 - Create list of Edge objects
        edges = [Edge(**edge_data) for edge_data in edges_data]
        # 返回GraphData实例 - Return GraphData instance
        return GraphData(nodes=nodes, edges=edges, metadata=metadata or {})

    @staticmethod
    def create_industry_chain_graph_data(nodes_data: List[Dict], edges_data: List[Dict], metadata: Optional[Dict] = None) -> GraphData:
        """
        创建产业链图的GraphData实例
        Create GraphData instance for industry chain graph

        Args:
            nodes_data (List[Dict]): 节点数据列表
            edges_data (List[Dict]): 边数据列表
            metadata (Optional[Dict]): 图的元数据

        Returns:
            GraphData: GraphData实例
        """
        # 创建节点对象列表 - Create list of Node objects
        nodes = [Node(**node_data) for node_data in nodes_data]
        # 创建边对象列表 - Create list of Edge objects
        edges = [Edge(**edge_data) for edge_data in edges_data]
        # 返回GraphData实例 - Return GraphData instance
        return GraphData(nodes=nodes, edges=edges, metadata=metadata or {})

    @staticmethod
    def _build_tree_nodes(tree_data_list: List[Dict]) -> List[TreeNode]:
        """
        递归构建树节点列表
        Recursively build list of TreeNode objects
        """
        # 初始化树节点列表 - Initialize list of TreeNode objects
        nodes = []
        # 遍历树数据列表 - Iterate through tree data list
        for item in tree_data_list:
            # 获取子节点数据 - Get children data
            children_data = item.pop("children", [])
            # 递归构建子树节点 - Recursively build child TreeNodes
            children_nodes = VisualizationDataFactory._build_tree_nodes(children_data) if children_data else []
            # 创建当前节点并添加到列表 - Create current node and add to list
            nodes.append(TreeNode(**item, children=children_nodes))
        # 返回构建的节点列表 - Return list of constructed nodes
        return nodes

    @staticmethod
    def create_stock_pool_tree_data(root_node_data: Dict, metadata: Optional[Dict] = None) -> TreeNode:
        """
        创建股票池分层树状图的TreeNode实例 (根节点)
        Create TreeNode instance (root node) for stock pool hierarchy tree

        Args:
            root_node_data (Dict): 根节点数据，包含id, name, value, children, properties
            metadata (Optional[Dict]): 树的元数据 (可以附加到根节点的properties中)

        Returns:
            TreeNode: TreeNode根节点实例
        """
        # 获取子节点数据 - Get children data
        children_data = root_node_data.pop("children", [])
        # 递归构建子树节点 - Recursively build child TreeNodes
        children_nodes = VisualizationDataFactory._build_tree_nodes(children_data) if children_data else []
        
        # 合并元数据到根节点属性 - Merge metadata into root node properties
        props = root_node_data.get("properties", {})
        if metadata:
            props.update(metadata) # 将元数据更新到属性中
        
        # 返回TreeNode根节点实例 - Return TreeNode root node instance
        return TreeNode(**root_node_data, children=children_nodes, properties=props)

    @staticmethod
    def create_flowchart_data(steps_data: List[Dict], connections_data: List[Dict[str,str]], metadata: Optional[Dict] = None) -> FlowchartData:
        """
        创建流程图的FlowchartData实例
        Create FlowchartData instance

        Args:
            steps_data (List[Dict]): 步骤数据列表，每个字典包含id, label, type, next_steps, properties
            connections_data (List[Dict[str,str]]): 连接数据列表，每个字典包含 from, to
            metadata (Optional[Dict]): 流程图的元数据

        Returns:
            FlowchartData: FlowchartData实例
        """
        # 创建步骤对象列表 - Create list of FlowchartStep objects
        steps = [FlowchartStep(**step_data) for step_data in steps_data]
        # 返回FlowchartData实例 - Return FlowchartData instance
        return FlowchartData(steps=steps, connections=connections_data, metadata=metadata or {})

if __name__ == '__main__':
    # 示例用法 - Example usage
    
    # 创建传导图数据 - Create transmission graph data
    nodes_trans = [
        {"id": "event", "label": "新闻事件", "type": "event"},
        {"id": "industryA", "label": "行业A", "type": "industry"},
        {"id": "companyX", "label": "公司X", "type": "company"}
    ]
    edges_trans = [
        {"source": "event", "target": "industryA", "label": "影响"},
        {"source": "industryA", "target": "companyX", "label": "传导"}
    ]
    transmission_graph = VisualizationDataFactory.create_transmission_graph_data(nodes_trans, edges_trans, {"name": "事件传导图"})
    print("传导图数据:") # 打印传导图数据
    print(transmission_graph.json(indent=2, ensure_ascii=False)) # 以JSON格式打印，缩进2个空格，不确保ASCII编码

    # 创建产业链图数据 - Create industry chain graph data
    nodes_chain = [
        {"id": "upstream1", "label": "上游环节1", "type": "upstream"},
        {"id": "midstream1", "label": "中游环节1", "type": "midstream"},
        {"id": "downstream1", "label": "下游环节1", "type": "downstream"}
    ]
    edges_chain = [
        {"source": "upstream1", "target": "midstream1", "label": "供应"},
        {"source": "midstream1", "target": "downstream1", "label": "供应"}
    ]
    industry_chain_graph = VisualizationDataFactory.create_industry_chain_graph_data(nodes_chain, edges_chain)
    print("\n产业链图数据:") # 打印产业链图数据
    print(industry_chain_graph.json(indent=2, ensure_ascii=False)) # 以JSON格式打印，缩进2个空格，不确保ASCII编码

    # 创建股票池树状图数据 - Create stock pool tree data
    stock_pool_tree_raw = {
        "id": "root", "name": "股票池总览", "value": 100,
        "children": [
            {"id": "high_rel", "name": "高相关性", "value": 30, 
             "children": [
                 {"id": "stockA", "name": "股票A", "value": 15},
                 {"id": "stockB", "name": "股票B", "value": 15}
             ]},
            {"id": "mid_rel", "name": "中相关性", "value": 70,
             "children": [
                 {"id": "stockC", "name": "股票C", "value": 40},
                 {"id": "stockD", "name": "股票D", "value": 30}
             ]}
        ]
    }
    stock_tree = VisualizationDataFactory.create_stock_pool_tree_data(stock_pool_tree_raw, {"description": "按相关性分层"})
    print("\n股票池树状图数据:") # 打印股票池树状图数据
    print(stock_tree.json(indent=2, ensure_ascii=False)) # 以JSON格式打印，缩进2个空格，不确保ASCII编码

    # 创建流程图数据 - Create flowchart data
    flow_steps = [
        {"id": "step1", "label": "开始", "type": "start", "next_steps": ["step2"]},
        {"id": "step2", "label": "处理A", "type": "process", "next_steps": ["step3"]},
        {"id": "step3", "label": "结束", "type": "end"}
    ]
    flow_connections = [
        {"from": "step1", "to": "step2"},
        {"from": "step2", "to": "step3"}
    ]
    flowchart = VisualizationDataFactory.create_flowchart_data(flow_steps, flow_connections)
    print("\n流程图数据:") # 打印流程图数据
    print(flowchart.json(indent=2, ensure_ascii=False)) # 以JSON格式打印，缩进2个空格，不确保ASCII编码
