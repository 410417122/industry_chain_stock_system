"""
产业链分析模块
Industry Chain Analysis Module

负责构建、维护和分析产业链知识图谱
Responsible for building, maintaining and analyzing industry chain knowledge graphs
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# 导入工具模块 - Import utility modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import LoggerMixin, log_execution_time
from utils.api_client import LLMClient

# 导入第三方库 - Import third-party libraries
try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False

class IndustryChainBuilder(LoggerMixin):
    """
    产业链构建器类
    Industry Chain Builder Class
    
    负责构建完整的产业链知识图谱和关系映射
    Responsible for building complete industry chain knowledge graphs and relationship mapping
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        初始化产业链构建器
        Initialize industry chain builder
        
        Args:
            llm_client: LLM客户端实例 - LLM client instance
        """
        self.llm_client = llm_client
        
        # 产业链层级定义 - Industry chain level definition
        self.chain_levels = {
            "upstream": "上游",
            "midstream": "中游", 
            "downstream": "下游",
            "related": "相关产业"
        }
        
        # 关系强度等级 - Relationship strength levels
        self.relationship_strength = {
            "strong": "强关联",
            "medium": "中等关联",
            "weak": "弱关联"
        }
        
        # 行业分类缓存 - Industry classification cache
        self.industry_cache = {}
        
        # 企业数据缓存 - Company data cache
        self.company_cache = {}
        
        self.logger.info("产业链构建器初始化完成")
    
    @log_execution_time
    def build_industry_chain(self, industry_name: str) -> Dict[str, Any]:
        """
        构建指定行业的完整产业链
        Build complete industry chain for specified industry
        
        Args:
            industry_name: 行业名称 - Industry name
            
        Returns:
            Dict[str, Any]: 产业链数据结构 - Industry chain data structure
        """
        self.logger.info(f"开始构建行业产业链: {industry_name}")
        
        try:
            # 第1步：获取行业基础信息 - Step 1: Get basic industry information
            industry_info = self._get_industry_info(industry_name)
            
            # 第2步：获取行业企业列表 - Step 2: Get industry company list
            companies = self._get_industry_companies(industry_name)
            
            # 第3步：构建产业链结构 - Step 3: Build industry chain structure
            chain_structure = self._build_chain_structure(industry_name, industry_info, companies)
            
            # 第4步：分析产业链关系 - Step 4: Analyze chain relationships
            relationships = self._analyze_chain_relationships(chain_structure)
            
            # 第5步：评估关系强度 - Step 5: Evaluate relationship strength
            relationship_scores = self._evaluate_relationship_strength(relationships)
            
            # 第6步：识别关键节点 - Step 6: Identify key nodes
            key_nodes = self._identify_key_nodes(chain_structure, relationship_scores)
            
            # 整合产业链数据 - Integrate industry chain data
            industry_chain = {
                "industry_name": industry_name,
                "build_timestamp": datetime.now().isoformat(),
                "basic_info": industry_info,
                "companies": companies,
                "chain_structure": chain_structure,
                "relationships": relationships,
                "relationship_scores": relationship_scores,
                "key_nodes": key_nodes,
                "metadata": {
                    "total_companies": len(companies),
                    "upstream_count": len(chain_structure.get("upstream", [])),
                    "midstream_count": len(chain_structure.get("midstream", [])),
                    "downstream_count": len(chain_structure.get("downstream", [])),
                    "related_count": len(chain_structure.get("related", []))
                }
            }
            
            self.logger.info(f"产业链构建完成: {industry_name}")
            return industry_chain
            
        except Exception as e:
            self.logger.error(f"产业链构建失败: {e}")
            return {
                "error": "产业链构建失败",
                "error_details": str(e),
                "industry_name": industry_name,
                "build_timestamp": datetime.now().isoformat()
            }
    
    def _get_industry_info(self, industry_name: str) -> Dict[str, Any]:
        """
        获取行业基础信息
        Get basic industry information
        
        Args:
            industry_name: 行业名称 - Industry name
            
        Returns:
            Dict[str, Any]: 行业基础信息 - Basic industry information
        """
        # 检查缓存 - Check cache
        if industry_name in self.industry_cache:
            self.logger.debug(f"从缓存获取行业信息: {industry_name}")
            return self.industry_cache[industry_name]
        
        # 构建行业信息获取提示 - Build industry info prompt
        prompt = f"""
        请提供关于"{industry_name}"行业的详细信息：

        请包含以下内容：
        1. 行业定义和主要特征
        2. 行业发展阶段和成熟度
        3. 主要产品和服务类型
        4. 行业规模和增长趋势
        5. 主要驱动因素
        6. 面临的主要挑战
        7. 政策环境和监管情况
        8. 技术发展趋势

        请以JSON格式返回：
        {{
            "definition": "行业定义",
            "characteristics": ["特征1", "特征2"],
            "development_stage": "发展阶段",
            "main_products": ["产品1", "产品2"],
            "market_size": "市场规模描述",
            "growth_trend": "增长趋势",
            "driving_factors": ["驱动因素1", "驱动因素2"],
            "challenges": ["挑战1", "挑战2"],
            "policy_environment": "政策环境描述",
            "technology_trends": ["技术趋势1", "技术趋势2"]
        }}
        """
        
        try:
            response = self.llm_client.generate_structured(prompt)
            
            if "error" not in response:
                industry_info = {
                    "definition": response.get("definition", "无定义"),
                    "characteristics": response.get("characteristics", []),
                    "development_stage": response.get("development_stage", "未知"),
                    "main_products": response.get("main_products", []),
                    "market_size": response.get("market_size", "未知"),
                    "growth_trend": response.get("growth_trend", "未知"),
                    "driving_factors": response.get("driving_factors", []),
                    "challenges": response.get("challenges", []),
                    "policy_environment": response.get("policy_environment", "未知"),
                    "technology_trends": response.get("technology_trends", [])
                }
                
                # 缓存结果 - Cache result
                self.industry_cache[industry_name] = industry_info
                return industry_info
            else:
                return self._fallback_industry_info()
                
        except Exception as e:
            self.logger.error(f"获取行业信息异常: {e}")
            return self._fallback_industry_info()
    
    def _get_industry_companies(self, industry_name: str) -> List[Dict[str, Any]]:
        """
        获取行业相关企业列表
        Get list of industry-related companies
        
        Args:
            industry_name: 行业名称 - Industry name
            
        Returns:
            List[Dict[str, Any]]: 企业列表 - List of companies
        """
        companies = []
        
        # 尝试使用AKshare获取实际数据 - Try to get real data using AKshare
        if HAS_AKSHARE:
            try:
                # 获取A股股票列表 - Get A-share stock list
                stock_list = ak.stock_info_a_code_name()
                
                # 根据行业名称筛选相关股票 - Filter related stocks by industry name
                industry_keywords = self._get_industry_keywords(industry_name)
                related_stocks = []
                
                for _, stock in stock_list.iterrows():
                    stock_name = str(stock.get('name', ''))
                    for keyword in industry_keywords:
                        if keyword in stock_name:
                            related_stocks.append({
                                "code": str(stock.get('code', '')),
                                "name": stock_name,
                                "source": "akshare"
                            })
                            break
                
                # 限制数量并去重 - Limit quantity and remove duplicates
                companies.extend(related_stocks[:20])
                
            except Exception as e:
                self.logger.warning(f"AKshare数据获取失败: {e}")
        
        # 使用LLM补充企业信息 - Use LLM to supplement company information
        llm_companies = self._get_companies_from_llm(industry_name)
        companies.extend(llm_companies)
        
        # 去重和清理 - Remove duplicates and clean
        unique_companies = self._deduplicate_companies(companies)
        
        return unique_companies[:30]  # 限制最大数量 - Limit maximum quantity
    
    def _get_industry_keywords(self, industry_name: str) -> List[str]:
        """
        获取行业关键词用于筛选
        Get industry keywords for filtering
        
        Args:
            industry_name: 行业名称 - Industry name
            
        Returns:
            List[str]: 关键词列表 - List of keywords
        """
        # 预定义关键词映射 - Predefined keyword mapping
        keyword_mapping = {
            "新能源": ["新能源", "锂电", "电池", "光伏", "风电", "储能"],
            "新能源汽车": ["新能源车", "电动车", "汽车", "动力电池"],
            "人工智能": ["人工智能", "AI", "智能", "机器人", "算法"],
            "半导体": ["半导体", "芯片", "集成电路", "电子", "晶圆"],
            "生物医药": ["生物", "医药", "制药", "疫苗", "医疗"],
            "5G通信": ["5G", "通信", "网络", "基站", "光纤"]
        }
        
        # 返回匹配的关键词或默认关键词 - Return matched keywords or default keywords
        return keyword_mapping.get(industry_name, [industry_name])
    
    def _get_companies_from_llm(self, industry_name: str) -> List[Dict[str, Any]]:
        """
        使用LLM获取企业信息
        Get company information using LLM
        
        Args:
            industry_name: 行业名称 - Industry name
            
        Returns:
            List[Dict[str, Any]]: 企业列表 - List of companies
        """
        prompt = f"""
        请列出"{industry_name}"行业的主要上市公司：

        请包含以下信息：
        1. 公司名称（全称和简称）
        2. 股票代码（如果知道）
        3. 主要业务描述
        4. 在产业链中的位置（上游/中游/下游）
        5. 市场地位（龙头/主要参与者/新兴企业）

        请列出15-20家有代表性的企业，以JSON格式返回：
        {{
            "companies": [
                {{
                    "name": "公司名称",
                    "code": "股票代码",
                    "business": "主要业务",
                    "position": "产业链位置",
                    "market_status": "市场地位"
                }}
            ]
        }}
        """
        
        try:
            response = self.llm_client.generate_structured(prompt)
            
            if "error" not in response and "companies" in response:
                companies = []
                for company in response["companies"]:
                    if isinstance(company, dict) and "name" in company:
                        companies.append({
                            "name": company.get("name", ""),
                            "code": company.get("code", ""),
                            "business": company.get("business", ""),
                            "position": company.get("position", ""),
                            "market_status": company.get("market_status", ""),
                            "source": "llm"
                        })
                return companies
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"LLM企业信息获取异常: {e}")
            return []
    
    def _build_chain_structure(
        self, 
        industry_name: str, 
        industry_info: Dict[str, Any], 
        companies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        构建产业链结构
        Build industry chain structure
        
        Args:
            industry_name: 行业名称 - Industry name
            industry_info: 行业信息 - Industry information
            companies: 企业列表 - Company list
            
        Returns:
            Dict[str, Any]: 产业链结构 - Industry chain structure
        """
        # 构建产业链结构分析提示 - Build chain structure analysis prompt
        prompt = f"""
        基于以下信息，构建"{industry_name}"的详细产业链结构：

        行业信息：
        - 定义：{industry_info.get('definition', '无')}
        - 主要产品：{', '.join(industry_info.get('main_products', []))}
        - 技术趋势：{', '.join(industry_info.get('technology_trends', []))}

        相关企业：{', '.join([c.get('name', '') for c in companies[:10]])}

        请详细分析产业链的各个环节：

        1. 上游（原材料、设备、技术供应）：
        - 主要环节和产品
        - 关键供应商类型
        - 技术壁垒和依赖关系

        2. 中游（制造、加工、集成）：
        - 核心制造环节
        - 主要参与者类型
        - 生产工艺和技术要求

        3. 下游（应用、销售、服务）：
        - 主要应用领域
        - 终端客户类型
        - 销售和服务模式

        4. 相关产业：
        - 配套产业
        - 替代产业
        - 互补产业

        请以JSON格式返回：
        {{
            "upstream": [
                {{
                    "segment": "环节名称",
                    "description": "详细描述",
                    "key_players": ["参与者类型1", "参与者类型2"],
                    "barriers": "技术壁垒",
                    "importance": "重要性等级"
                }}
            ],
            "midstream": [...],
            "downstream": [...],
            "related": [...]
        }}
        """
        
        try:
            response = self.llm_client.generate_structured(prompt)
            
            if "error" not in response:
                # 验证和清理结构 - Validate and clean structure
                chain_structure = {}
                for level in ["upstream", "midstream", "downstream", "related"]:
                    if level in response and isinstance(response[level], list):
                        chain_structure[level] = []
                        for segment in response[level]:
                            if isinstance(segment, dict):
                                chain_structure[level].append({
                                    "segment": segment.get("segment", "未知环节"),
                                    "description": segment.get("description", "无描述"),
                                    "key_players": segment.get("key_players", []),
                                    "barriers": segment.get("barriers", "无"),
                                    "importance": segment.get("importance", "中")
                                })
                    else:
                        chain_structure[level] = []
                
                return chain_structure
            else:
                return self._fallback_chain_structure()
                
        except Exception as e:
            self.logger.error(f"产业链结构构建异常: {e}")
            return self._fallback_chain_structure()
    
    def _analyze_chain_relationships(self, chain_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析产业链关系
        Analyze industry chain relationships
        
        Args:
            chain_structure: 产业链结构 - Industry chain structure
            
        Returns:
            Dict[str, Any]: 关系分析结果 - Relationship analysis result
        """
        prompt = f"""
        基于以下产业链结构，分析各环节之间的关系：

        产业链结构：
        {json.dumps(chain_structure, ensure_ascii=False, indent=2)}

        请分析：
        1. 上游→中游的供应关系
        2. 中游→下游的供应关系
        3. 各环节的依赖强度
        4. 替代关系和竞争关系
        5. 协同效应和互补关系
        6. 关键控制点和瓶颈

        请以JSON格式返回：
        {{
            "supply_relationships": [
                {{
                    "from": "供应方",
                    "to": "需求方",
                    "relationship_type": "关系类型",
                    "strength": "关系强度",
                    "description": "关系描述"
                }}
            ],
            "dependency_analysis": {{
                "high_dependency": ["高依赖环节"],
                "medium_dependency": ["中依赖环节"],
                "low_dependency": ["低依赖环节"]
            }},
            "bottlenecks": ["瓶颈点1", "瓶颈点2"],
            "synergies": ["协同点1", "协同点2"]
        }}
        """
        
        try:
            response = self.llm_client.generate_structured(prompt)
            
            if "error" not in response:
                return {
                    "supply_relationships": response.get("supply_relationships", []),
                    "dependency_analysis": response.get("dependency_analysis", {}),
                    "bottlenecks": response.get("bottlenecks", []),
                    "synergies": response.get("synergies", [])
                }
            else:
                return self._fallback_relationships()
                
        except Exception as e:
            self.logger.error(f"关系分析异常: {e}")
            return self._fallback_relationships()
    
    def _evaluate_relationship_strength(self, relationships: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估关系强度
        Evaluate relationship strength
        
        Args:
            relationships: 关系数据 - Relationship data
            
        Returns:
            Dict[str, Any]: 关系强度评估 - Relationship strength evaluation
        """
        # 简化的关系强度评估 - Simplified relationship strength evaluation
        supply_rels = relationships.get("supply_relationships", [])
        
        strength_scores = {}
        for rel in supply_rels:
            from_node = rel.get("from", "")
            to_node = rel.get("to", "")
            strength = rel.get("strength", "medium")
            
            # 转换强度为数值 - Convert strength to numerical value
            strength_value = {
                "high": 0.9,
                "strong": 0.9,
                "medium": 0.6,
                "low": 0.3,
                "weak": 0.3
            }.get(strength.lower(), 0.5)
            
            relationship_key = f"{from_node}->{to_node}"
            strength_scores[relationship_key] = {
                "strength_value": strength_value,
                "strength_label": strength,
                "relationship_type": rel.get("relationship_type", "供应关系")
            }
        
        return {
            "relationship_scores": strength_scores,
            "average_strength": sum(s["strength_value"] for s in strength_scores.values()) / len(strength_scores) if strength_scores else 0,
            "strong_relationships": [k for k, v in strength_scores.items() if v["strength_value"] > 0.7],
            "weak_relationships": [k for k, v in strength_scores.items() if v["strength_value"] < 0.5]
        }
    
    def _identify_key_nodes(
        self, 
        chain_structure: Dict[str, Any], 
        relationship_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        识别关键节点
        Identify key nodes
        
        Args:
            chain_structure: 产业链结构 - Industry chain structure
            relationship_scores: 关系强度评分 - Relationship strength scores
            
        Returns:
            Dict[str, Any]: 关键节点分析 - Key nodes analysis
        """
        key_nodes = {
            "core_nodes": [],
            "control_points": [],
            "innovation_nodes": [],
            "bottleneck_nodes": []
        }
        
        # 分析各层级的重要节点 - Analyze important nodes in each level
        for level, segments in chain_structure.items():
            if level in ["upstream", "midstream", "downstream"]:
                for segment in segments:
                    importance = segment.get("importance", "medium")
                    barriers = segment.get("barriers", "")
                    
                    node_info = {
                        "name": segment.get("segment", ""),
                        "level": level,
                        "description": segment.get("description", ""),
                        "importance": importance
                    }
                    
                    # 根据重要性和特征分类 - Classify by importance and characteristics
                    if importance.lower() in ["high", "高", "核心"]:
                        key_nodes["core_nodes"].append(node_info)
                    
                    if "技术" in barriers or "专利" in barriers:
                        key_nodes["innovation_nodes"].append(node_info)
                    
                    if "瓶颈" in barriers or "稀缺" in barriers:
                        key_nodes["bottleneck_nodes"].append(node_info)
        
        # 基于关系强度识别控制点 - Identify control points based on relationship strength
        strong_rels = relationship_scores.get("strong_relationships", [])
        for rel in strong_rels:
            from_node = rel.split("->")[0] if "->" in rel else rel
            key_nodes["control_points"].append({
                "name": from_node,
                "type": "supply_control",
                "description": "强供应关系控制点"
            })
        
        return key_nodes
    
    def analyze_chain_impact(
        self, 
        industry_chain: Dict[str, Any], 
        event_description: str
    ) -> Dict[str, Any]:
        """
        分析事件对产业链的影响
        Analyze event impact on industry chain
        
        Args:
            industry_chain: 产业链数据 - Industry chain data
            event_description: 事件描述 - Event description
            
        Returns:
            Dict[str, Any]: 影响分析结果 - Impact analysis result
        """
        prompt = f"""
        基于以下产业链结构和事件描述，分析事件对产业链各环节的具体影响：

        事件描述：{event_description}

        产业链结构：
        {json.dumps(industry_chain.get('chain_structure', {}), ensure_ascii=False, indent=2)}

        关键节点：
        {json.dumps(industry_chain.get('key_nodes', {}), ensure_ascii=False, indent=2)}

        请详细分析：
        1. 对上游环节的影响（供应、成本、技术等）
        2. 对中游环节的影响（生产、竞争、市场等）
        3. 对下游环节的影响（需求、价格、渠道等）
        4. 影响传导路径和时间窗口
        5. 受益最大的环节和企业类型
        6. 受损最大的环节和企业类型

        请以JSON格式返回：
        {{
            "upstream_impact": {{
                "description": "上游影响描述",
                "affected_segments": ["受影响环节"],
                "impact_type": "positive/negative/mixed",
                "impact_intensity": "high/medium/low"
            }},
            "midstream_impact": {{...}},
            "downstream_impact": {{...}},
            "transmission_path": ["传导路径步骤"],
            "time_windows": {{
                "immediate": "即时影响",
                "short_term": "短期影响",
                "long_term": "长期影响"
            }},
            "beneficiaries": ["受益类型"],
            "affected_parties": ["受损类型"]
        }}
        """
        
        try:
            response = self.llm_client.generate_structured(prompt)
            
            if "error" not in response:
                return {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "event_description": event_description,
                    "impact_analysis": response
                }
            else:
                return self._fallback_impact_analysis()
                
        except Exception as e:
            self.logger.error(f"产业链影响分析异常: {e}")
            return self._fallback_impact_analysis()
    
    # 辅助方法 - Helper methods
    def _deduplicate_companies(self, companies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重企业列表 - Deduplicate company list"""
        seen_names = set()
        unique_companies = []
        
        for company in companies:
            name = company.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                unique_companies.append(company)
        
        return unique_companies
    
    def _fallback_industry_info(self) -> Dict[str, Any]:
        """行业信息获取失败时的后备方案 - Fallback for industry info failure"""
        return {
            "definition": "信息获取失败",
            "characteristics": [],
            "development_stage": "未知",
            "main_products": [],
            "market_size": "未知",
            "growth_trend": "未知",
            "driving_factors": [],
            "challenges": [],
            "policy_environment": "未知",
            "technology_trends": []
        }
    
    def _fallback_chain_structure(self) -> Dict[str, Any]:
        """产业链结构构建失败时的后备方案 - Fallback for chain structure failure"""
        return {
            "upstream": [],
            "midstream": [],
            "downstream": [],
            "related": []
        }
    
    def _fallback_relationships(self) -> Dict[str, Any]:
        """关系分析失败时的后备方案 - Fallback for relationship analysis failure"""
        return {
            "supply_relationships": [],
            "dependency_analysis": {},
            "bottlenecks": [],
            "synergies": []
        }
    
    def _fallback_impact_analysis(self) -> Dict[str, Any]:
        """影响分析失败时的后备方案 - Fallback for impact analysis failure"""
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "event_description": "分析失败",
            "impact_analysis": {
                "error": "影响分析失败",
                "upstream_impact": {"description": "分析失败", "impact_type": "unknown"},
                "midstream_impact": {"description": "分析失败", "impact_type": "unknown"},
                "downstream_impact": {"description": "分析失败", "impact_type": "unknown"}
            }
        }
    
    def get_cached_industries(self) -> List[str]:
        """
        获取已缓存的行业列表
        Get list of cached industries
        
        Returns:
            List[str]: 缓存的行业名称列表 - List of cached industry names
        """
        return list(self.industry_cache.keys())
    
    def clear_cache(self):
        """清除缓存 - Clear cache"""
        self.industry_cache.clear()
        self.company_cache.clear()
        self.logger.info("产业链缓存已清除")

if __name__ == "__main__":
    # 测试产业链构建器 - Test industry chain builder
    from utils.api_client import create_default_client
    
    print("测试产业链构建器...")
    
    # 创建LLM客户端和产业链构建器 - Create LLM client and industry chain builder
    llm_client = create_default_client()
    builder = IndustryChainBuilder(llm_client)
    
    # 测试行业 - Test industry
    test_industry = "新能源汽车"
    
    # 构建产业链 - Build industry chain
    try:
        chain = builder.build_industry_chain(test_industry)
        
        if "error" not in chain:
            print("✓ 产业链构建成功")
            print(f"行业名称: {chain.get('industry_name', 'N/A')}")
            print(f"企业数量: {chain.get('metadata', {}).get('total_companies', 0)}")
            print(f"上游环节: {chain.get('metadata', {}).get('upstream_count', 0)}")
            print(f"中游环节: {chain.get('metadata', {}).get('midstream_count', 0)}")
            print(f"下游环节: {chain.get('metadata', {}).get('downstream_count', 0)}")
            
            # 测试影响分析 - Test impact analysis
            test_event = "政府发布新能源汽车补贴政策"
            impact = builder.analyze_chain_impact(chain, test_event)
            print(f"影响分析: {impact.get('impact_analysis', {}).get('upstream_impact', {}).get('description', 'N/A')[:50]}...")
            
        else:
            print(f"✗ 产业链构建失败: {chain.get('error_details', 'N/A')}")
        
    except Exception as e:
        print(f"✗ 测试异常: {e}")
    
    print("测试完成")
