"""
因果推理模块
Causal Reasoning Module

负责构建和执行复杂的因果推理链条，分析事件对产业链的影响传导机制
Responsible for building and executing complex causal reasoning chains, analyzing event impact transmission mechanisms on industry chains
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

# 导入工具模块 - Import utility modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import LoggerMixin, log_execution_time
from utils.api_client import LLMClient

class CausalReasoningEngine(LoggerMixin):
    """
    因果推理引擎类
    Causal Reasoning Engine Class
    
    负责构建事件影响的因果图和推理链条
    Responsible for building causal graphs and reasoning chains for event impacts
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        初始化因果推理引擎
        Initialize causal reasoning engine
        
        Args:
            llm_client: LLM客户端实例 - LLM client instance
        """
        self.llm_client = llm_client
        
        # 因果关系类型定义 - Causal relationship type definitions
        self.causal_types = {
            "direct": "直接因果",
            "indirect": "间接因果",
            "mediating": "中介因果",
            "moderating": "调节因果",
            "confounding": "混杂因果"
        }
        
        # 推理强度等级 - Reasoning strength levels
        self.reasoning_strength = {
            "strong": {"value": 0.8, "label": "强推理"},
            "medium": {"value": 0.6, "label": "中等推理"},
            "weak": {"value": 0.4, "label": "弱推理"},
            "speculative": {"value": 0.2, "label": "推测性"}
        }
        
        # 时间窗口定义 - Time window definitions
        self.time_windows = {
            "immediate": {"days": 1, "label": "即时影响"},
            "short_term": {"days": 30, "label": "短期影响"},
            "medium_term": {"days": 180, "label": "中期影响"},
            "long_term": {"days": 365, "label": "长期影响"}
        }
        
        # 推理历史缓存 - Reasoning history cache
        self.reasoning_cache = {}
        
        self.logger.info("因果推理引擎初始化完成")
    
    @log_execution_time
    def create_causal_graph(
        self, 
        event_info: Dict[str, Any], 
        industry_chain: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        创建事件影响的因果图
        Create causal graph for event impact
        
        Args:
            event_info: 事件信息 - Event information
            industry_chain: 产业链数据 - Industry chain data
            
        Returns:
            Dict[str, Any]: 因果图数据 - Causal graph data
        """
        self.logger.info("开始构建因果推理图")
        
        try:
            # 第1步：分析直接因果关系 - Step 1: Analyze direct causal relationships
            direct_causality = self._analyze_direct_causality(event_info, industry_chain)
            
            # 第2步：分析间接因果关系 - Step 2: Analyze indirect causal relationships
            indirect_causality = self._analyze_indirect_causality(direct_causality, industry_chain)
            
            # 第3步：构建传导路径 - Step 3: Build transmission paths
            transmission_paths = self._build_transmission_paths(direct_causality, indirect_causality)
            
            # 第4步：评估因果强度 - Step 4: Evaluate causal strength
            causal_strength = self._evaluate_causal_strength(transmission_paths, event_info)
            
            # 第5步：分析时间动态 - Step 5: Analyze temporal dynamics
            temporal_analysis = self._analyze_temporal_dynamics(transmission_paths, event_info)
            
            # 第6步：识别关键假设 - Step 6: Identify key assumptions
            key_assumptions = self._identify_key_assumptions(transmission_paths, event_info)
            
            # 第7步：进行不确定性分析 - Step 7: Conduct uncertainty analysis
            uncertainty_analysis = self._conduct_uncertainty_analysis(transmission_paths, causal_strength)
            
            # 整合因果图数据 - Integrate causal graph data
            causal_graph = {
                "event_info": event_info,
                "industry_chain_name": industry_chain.get("industry_name", "未知"),
                "creation_timestamp": datetime.now().isoformat(),
                "direct_causality": direct_causality,
                "indirect_causality": indirect_causality,
                "transmission_paths": transmission_paths,
                "causal_strength": causal_strength,
                "temporal_analysis": temporal_analysis,
                "key_assumptions": key_assumptions,
                "uncertainty_analysis": uncertainty_analysis,
                "reasoning_metadata": {
                    "total_paths": len(transmission_paths),
                    "high_confidence_paths": len([p for p in transmission_paths if p.get("confidence", 0) > 0.7]),
                    "average_path_length": sum(len(p.get("steps", [])) for p in transmission_paths) / len(transmission_paths) if transmission_paths else 0
                }
            }
            
            self.logger.info("因果推理图构建完成")
            return causal_graph
            
        except Exception as e:
            self.logger.error(f"因果推理图构建失败: {e}")
            return {
                "error": "因果推理图构建失败",
                "error_details": str(e),
                "event_info": event_info,
                "creation_timestamp": datetime.now().isoformat()
            }
    
    def _analyze_direct_causality(
        self, 
        event_info: Dict[str, Any], 
        industry_chain: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        分析直接因果关系
        Analyze direct causal relationships
        
        Args:
            event_info: 事件信息 - Event information
            industry_chain: 产业链数据 - Industry chain data
            
        Returns:
            Dict[str, Any]: 直接因果关系分析 - Direct causal relationship analysis
        """
        # 构建直接因果分析提示 - Build direct causal analysis prompt
        prompt = f"""
        基于以下事件信息和产业链结构，分析事件的直接因果影响：

        事件信息：
        - 事件描述：{event_info.get('basic_info', {}).get('event_description', '未知事件')}
        - 事件类型：{event_info.get('event_classification', {}).get('event_type_cn', '未知类型')}
        - 影响方向：{event_info.get('event_classification', {}).get('impact_direction', '未知')}
        - 主要实体：{', '.join(event_info.get('entities', {}).get('companies', []))}

        产业链结构：
        - 上游环节：{len(industry_chain.get('chain_structure', {}).get('upstream', []))}个环节
        - 中游环节：{len(industry_chain.get('chain_structure', {}).get('midstream', []))}个环节
        - 下游环节：{len(industry_chain.get('chain_structure', {}).get('downstream', []))}个环节

        请分析事件的直接因果影响：

        1. 直接受影响的产业链环节
        2. 影响机制和作用路径
        3. 影响强度和确定性
        4. 立即可观察的效应
        5. 直接受影响的企业类型

        请以JSON格式返回：
        {{
            "directly_affected_segments": [
                {{
                    "segment_name": "环节名称",
                    "segment_level": "upstream/midstream/downstream",
                    "impact_mechanism": "影响机制描述",
                    "impact_type": "positive/negative/mixed",
                    "impact_intensity": "high/medium/low",
                    "confidence_level": "high/medium/low",
                    "immediate_effects": ["效应1", "效应2"]
                }}
            ],
            "affected_company_types": ["企业类型1", "企业类型2"],
            "causal_mechanisms": [
                {{
                    "mechanism": "机制描述",
                    "explanation": "详细解释",
                    "evidence_strength": "strong/medium/weak"
                }}
            ]
        }}
        """
        
        try:
            response = self.llm_client.generate_structured(prompt)
            
            if "error" not in response:
                return {
                    "directly_affected_segments": response.get("directly_affected_segments", []),
                    "affected_company_types": response.get("affected_company_types", []),
                    "causal_mechanisms": response.get("causal_mechanisms", []),
                    "analysis_confidence": self._calculate_analysis_confidence(response)
                }
            else:
                return self._fallback_direct_causality()
                
        except Exception as e:
            self.logger.error(f"直接因果分析异常: {e}")
            return self._fallback_direct_causality()
    
    def _analyze_indirect_causality(
        self, 
        direct_causality: Dict[str, Any], 
        industry_chain: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        分析间接因果关系
        Analyze indirect causal relationships
        
        Args:
            direct_causality: 直接因果关系 - Direct causal relationships
            industry_chain: 产业链数据 - Industry chain data
            
        Returns:
            Dict[str, Any]: 间接因果关系分析 - Indirect causal relationship analysis
        """
        # 构建间接因果分析提示 - Build indirect causal analysis prompt
        prompt = f"""
        基于以下直接因果分析和产业链数据，分析间接因果关系和传导效应：

        直接因果分析：
        {json.dumps(direct_causality, ensure_ascii=False, indent=2)}

        产业链关系：
        {json.dumps(industry_chain.get('relationships', {}), ensure_ascii=False, indent=2)}

        请分析间接因果关系：

        1. 二级传导效应（直接影响的连锁反应）
        2. 三级传导效应（更深层的间接影响）
        3. 跨产业链的溢出效应
        4. 反馈循环和相互作用
        5. 累积效应和时间延迟

        请以JSON格式返回：
        {{
            "second_order_effects": [
                {{
                    "trigger_segment": "触发环节",
                    "affected_segment": "受影响环节", 
                    "transmission_mechanism": "传导机制",
                    "impact_description": "影响描述",
                    "time_delay": "时间延迟",
                    "confidence": "confidence_score"
                }}
            ],
            "third_order_effects": [...],
            "spillover_effects": [
                {{
                    "source_industry": "源产业",
                    "target_industry": "目标产业",
                    "spillover_mechanism": "溢出机制",
                    "impact_magnitude": "影响程度"
                }}
            ],
            "feedback_loops": [
                {{
                    "loop_description": "反馈循环描述",
                    "loop_type": "positive/negative",
                    "loop_strength": "强度评估"
                }}
            ]
        }}
        """
        
        try:
            response = self.llm_client.generate_structured(prompt)
            
            if "error" not in response:
                return {
                    "second_order_effects": response.get("second_order_effects", []),
                    "third_order_effects": response.get("third_order_effects", []),
                    "spillover_effects": response.get("spillover_effects", []),
                    "feedback_loops": response.get("feedback_loops", []),
                    "transmission_complexity": self._assess_transmission_complexity(response)
                }
            else:
                return self._fallback_indirect_causality()
                
        except Exception as e:
            self.logger.error(f"间接因果分析异常: {e}")
            return self._fallback_indirect_causality()
    
    def _build_transmission_paths(
        self, 
        direct_causality: Dict[str, Any], 
        indirect_causality: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        构建完整的传导路径
        Build complete transmission paths
        
        Args:
            direct_causality: 直接因果关系 - Direct causal relationships
            indirect_causality: 间接因果关系 - Indirect causal relationships
            
        Returns:
            List[Dict[str, Any]]: 传导路径列表 - List of transmission paths
        """
        transmission_paths = []
        
        # 基于直接影响构建一级路径 - Build first-level paths based on direct impacts
        for segment in direct_causality.get("directly_affected_segments", []):
            path = {
                "path_id": f"direct_{len(transmission_paths)}",
                "path_type": "direct",
                "path_length": 1,
                "steps": [
                    {
                        "step_number": 1,
                        "step_type": "direct_impact",
                        "from_element": "触发事件",
                        "to_element": segment.get("segment_name", "未知环节"),
                        "mechanism": segment.get("impact_mechanism", "未知机制"),
                        "impact_type": segment.get("impact_type", "unknown"),
                        "confidence": self._map_confidence_to_score(segment.get("confidence_level", "medium"))
                    }
                ],
                "overall_impact": segment.get("impact_type", "unknown"),
                "confidence": self._map_confidence_to_score(segment.get("confidence_level", "medium")),
                "time_window": "immediate"
            }
            transmission_paths.append(path)
        
        # 基于间接影响构建多级路径 - Build multi-level paths based on indirect impacts
        for effect in indirect_causality.get("second_order_effects", []):
            path = {
                "path_id": f"indirect2_{len(transmission_paths)}",
                "path_type": "indirect",
                "path_length": 2,
                "steps": [
                    {
                        "step_number": 1,
                        "step_type": "direct_impact",
                        "from_element": "触发事件",
                        "to_element": effect.get("trigger_segment", "未知"),
                        "mechanism": "直接影响",
                        "confidence": 0.7
                    },
                    {
                        "step_number": 2,
                        "step_type": "transmission",
                        "from_element": effect.get("trigger_segment", "未知"),
                        "to_element": effect.get("affected_segment", "未知"),
                        "mechanism": effect.get("transmission_mechanism", "未知机制"),
                        "confidence": float(effect.get("confidence", 0.6))
                    }
                ],
                "overall_impact": "mixed",  # 间接影响通常更复杂
                "confidence": float(effect.get("confidence", 0.6)),
                "time_window": self._determine_time_window(effect.get("time_delay", "short"))
            }
            transmission_paths.append(path)
        
        # 基于三级效应构建长路径 - Build long paths based on third-order effects
        for effect in indirect_causality.get("third_order_effects", []):
            path = {
                "path_id": f"indirect3_{len(transmission_paths)}",
                "path_type": "long_indirect",
                "path_length": 3,
                "steps": [
                    {
                        "step_number": 1,
                        "step_type": "direct_impact",
                        "from_element": "触发事件",
                        "to_element": "一级影响",
                        "mechanism": "直接影响",
                        "confidence": 0.7
                    },
                    {
                        "step_number": 2,
                        "step_type": "transmission", 
                        "from_element": "一级影响",
                        "to_element": "二级影响",
                        "mechanism": "产业链传导",
                        "confidence": 0.6
                    },
                    {
                        "step_number": 3,
                        "step_type": "transmission",
                        "from_element": "二级影响",
                        "to_element": effect.get("affected_segment", "未知"),
                        "mechanism": effect.get("transmission_mechanism", "未知机制"),
                        "confidence": float(effect.get("confidence", 0.5))
                    }
                ],
                "overall_impact": "speculative",
                "confidence": float(effect.get("confidence", 0.5)),
                "time_window": "long_term"
            }
            transmission_paths.append(path)
        
        return transmission_paths
    
    def _evaluate_causal_strength(
        self, 
        transmission_paths: List[Dict[str, Any]], 
        event_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        评估因果强度
        Evaluate causal strength
        
        Args:
            transmission_paths: 传导路径 - Transmission paths
            event_info: 事件信息 - Event information
            
        Returns:
            Dict[str, Any]: 因果强度评估 - Causal strength evaluation
        """
        strength_evaluation = {
            "overall_strength": 0.0,
            "path_strengths": [],
            "strength_distribution": {"strong": 0, "medium": 0, "weak": 0},
            "confidence_weighted_strength": 0.0
        }
        
        total_confidence = 0.0
        weighted_strength_sum = 0.0
        
        for path in transmission_paths:
            path_confidence = path.get("confidence", 0.5)
            path_length = path.get("path_length", 1)
            
            # 路径强度随长度衰减 - Path strength decays with length
            base_strength = path_confidence * (0.8 ** (path_length - 1))
            
            # 根据事件类型调整强度 - Adjust strength based on event type
            event_type = event_info.get("event_classification", {}).get("event_type", "unknown")
            if event_type == "policy":
                base_strength *= 1.2  # 政策事件通常影响更强
            elif event_type == "technology":
                base_strength *= 1.1  # 技术事件影响适中
            elif event_type == "market":
                base_strength *= 1.0  # 市场事件影响标准
            
            # 限制强度范围 - Limit strength range
            final_strength = min(base_strength, 1.0)
            
            path_strength_info = {
                "path_id": path.get("path_id", "unknown"),
                "path_type": path.get("path_type", "unknown"),
                "strength_score": final_strength,
                "strength_category": self._categorize_strength(final_strength),
                "confidence": path_confidence
            }
            
            strength_evaluation["path_strengths"].append(path_strength_info)
            
            # 更新分布统计 - Update distribution statistics
            category = self._categorize_strength(final_strength)
            strength_evaluation["strength_distribution"][category] += 1
            
            # 计算置信度加权强度 - Calculate confidence-weighted strength
            total_confidence += path_confidence
            weighted_strength_sum += final_strength * path_confidence
        
        # 计算总体强度 - Calculate overall strength
        if transmission_paths:
            strength_evaluation["overall_strength"] = sum(
                p["strength_score"] for p in strength_evaluation["path_strengths"]
            ) / len(transmission_paths)
            
            if total_confidence > 0:
                strength_evaluation["confidence_weighted_strength"] = weighted_strength_sum / total_confidence
        
        return strength_evaluation
    
    def _analyze_temporal_dynamics(
        self, 
        transmission_paths: List[Dict[str, Any]], 
        event_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        分析时间动态
        Analyze temporal dynamics
        
        Args:
            transmission_paths: 传导路径 - Transmission paths
            event_info: 事件信息 - Event information
            
        Returns:
            Dict[str, Any]: 时间动态分析 - Temporal dynamics analysis
        """
        temporal_analysis = {
            "time_windows": {},
            "peak_impact_time": "unknown",
            "duration_estimate": "unknown",
            "temporal_pattern": "unknown"
        }
        
        # 分析各时间窗口的影响 - Analyze impact in each time window
        for window_name, window_info in self.time_windows.items():
            relevant_paths = [
                p for p in transmission_paths 
                if p.get("time_window") == window_name
            ]
            
            if relevant_paths:
                avg_impact = sum(p.get("confidence", 0) for p in relevant_paths) / len(relevant_paths)
                temporal_analysis["time_windows"][window_name] = {
                    "path_count": len(relevant_paths),
                    "average_impact": avg_impact,
                    "impact_types": list(set(p.get("overall_impact") for p in relevant_paths)),
                    "description": window_info["label"]
                }
        
        # 确定峰值影响时间 - Determine peak impact time
        if temporal_analysis["time_windows"]:
            peak_window = max(
                temporal_analysis["time_windows"].items(),
                key=lambda x: x[1]["average_impact"]
            )
            temporal_analysis["peak_impact_time"] = peak_window[0]
        
        # 分析时间模式 - Analyze temporal pattern
        temporal_analysis["temporal_pattern"] = self._identify_temporal_pattern(transmission_paths)
        
        return temporal_analysis
    
    def _identify_key_assumptions(
        self, 
        transmission_paths: List[Dict[str, Any]], 
        event_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        识别关键假设
        Identify key assumptions
        
        Args:
            transmission_paths: 传导路径 - Transmission paths
            event_info: 事件信息 - Event information
            
        Returns:
            List[Dict[str, Any]]: 关键假设列表 - List of key assumptions
        """
        # 构建假设识别提示 - Build assumption identification prompt
        prompt = f"""
        基于以下因果传导路径分析，识别关键假设条件：

        事件类型：{event_info.get('event_classification', {}).get('event_type_cn', '未知')}
        传导路径数量：{len(transmission_paths)}
        主要影响方向：{event_info.get('event_classification', {}).get('impact_direction', '未知')}

        请识别以下类型的关键假设：

        1. 市场假设（市场反应、需求变化等）
        2. 政策假设（政策持续性、执行力度等）
        3. 技术假设（技术可行性、采用速度等）
        4. 竞争假设（竞争格局、企业反应等）
        5. 宏观假设（经济环境、资金流动等）

        请以JSON格式返回：
        {{
            "key_assumptions": [
                {{
                    "assumption_type": "假设类型",
                    "assumption_content": "假设内容",
                    "importance": "high/medium/low",
                    "validity_risk": "风险评估",
                    "impact_if_violated": "违反假设的影响"
                }}
            ]
        }}
        """
        
        try:
            response = self.llm_client.generate_structured(prompt)
            
            if "error" not in response and "key_assumptions" in response:
                return response["key_assumptions"]
            else:
                return self._fallback_key_assumptions()
                
        except Exception as e:
            self.logger.error(f"关键假设识别异常: {e}")
            return self._fallback_key_assumptions()
    
    def _conduct_uncertainty_analysis(
        self, 
        transmission_paths: List[Dict[str, Any]], 
        causal_strength: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        进行不确定性分析
        Conduct uncertainty analysis
        
        Args:
            transmission_paths: 传导路径 - Transmission paths
            causal_strength: 因果强度 - Causal strength
            
        Returns:
            Dict[str, Any]: 不确定性分析 - Uncertainty analysis
        """
        uncertainty_analysis = {
            "overall_uncertainty": 0.0,
            "uncertainty_sources": [],
            "confidence_intervals": {},
            "sensitivity_analysis": {}
        }
        
        # 计算总体不确定性 - Calculate overall uncertainty
        path_confidences = [p.get("confidence", 0.5) for p in transmission_paths]
        if path_confidences:
            avg_confidence = sum(path_confidences) / len(path_confidences)
            uncertainty_analysis["overall_uncertainty"] = 1.0 - avg_confidence
        
        # 识别不确定性来源 - Identify uncertainty sources
        uncertainty_sources = []
        
        # 路径长度相关的不确定性 - Path length related uncertainty
        long_paths = [p for p in transmission_paths if p.get("path_length", 1) > 2]
        if long_paths:
            uncertainty_sources.append({
                "source": "路径复杂性",
                "description": f"{len(long_paths)}个长路径增加了预测不确定性",
                "impact": "medium"
            })
        
        # 低置信度路径的不确定性 - Low confidence path uncertainty
        low_conf_paths = [p for p in transmission_paths if p.get("confidence", 1.0) < 0.5]
        if low_conf_paths:
            uncertainty_sources.append({
                "source": "推理置信度",
                "description": f"{len(low_conf_paths)}个低置信度路径",
                "impact": "high"
            })
        
        # 强度分布的不确定性 - Strength distribution uncertainty
        strength_dist = causal_strength.get("strength_distribution", {})
        weak_ratio = strength_dist.get("weak", 0) / max(sum(strength_dist.values()), 1)
        if weak_ratio > 0.5:
            uncertainty_sources.append({
                "source": "因果强度",
                "description": "超过50%的路径为弱因果关系",
                "impact": "medium"
            })
        
        uncertainty_analysis["uncertainty_sources"] = uncertainty_sources
        
        return uncertainty_analysis
    
    def validate_reasoning_chain(
        self, 
        causal_graph: Dict[str, Any], 
        validation_criteria: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        验证推理链的逻辑一致性
        Validate logical consistency of reasoning chain
        
        Args:
            causal_graph: 因果图数据 - Causal graph data
            validation_criteria: 验证标准 - Validation criteria
            
        Returns:
            Dict[str, Any]: 验证结果 - Validation result
        """
        validation_result = {
            "is_valid": False,
            "validation_score": 0.0,
            "validation_details": {},
            "recommendations": []
        }
        
        try:
            # 验证逻辑一致性 - Validate logical consistency
            logic_score = self._validate_logic_consistency(causal_graph)
            
            # 验证因果强度合理性 - Validate causal strength reasonableness
            strength_score = self._validate_strength_reasonableness(causal_graph)
            
            # 验证时间一致性 - Validate temporal consistency
            temporal_score = self._validate_temporal_consistency(causal_graph)
            
            # 验证假设合理性 - Validate assumption reasonableness
            assumption_score = self._validate_assumption_reasonableness(causal_graph)
            
            # 计算总体验证分数 - Calculate overall validation score
            validation_result["validation_score"] = (
                logic_score * 0.3 + 
                strength_score * 0.25 + 
                temporal_score * 0.25 + 
                assumption_score * 0.2
            )
            
            validation_result["is_valid"] = validation_result["validation_score"] > 0.6
            
            validation_result["validation_details"] = {
                "logic_consistency": logic_score,
                "strength_reasonableness": strength_score,
                "temporal_consistency": temporal_score,
                "assumption_reasonableness": assumption_score
            }
            
            # 生成改进建议 - Generate improvement recommendations
            if validation_result["validation_score"] < 0.8:
                validation_result["recommendations"] = self._generate_improvement_recommendations(
                    validation_result["validation_details"]
                )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"推理链验证异常: {e}")
            return {
                "is_valid": False,
                "validation_score": 0.0,
                "error": "验证过程异常",
                "error_details": str(e)
            }
    
    # 辅助方法 - Helper methods
    def _calculate_analysis_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """计算分析置信度 - Calculate analysis confidence"""
        # 基于分析结果的完整性和详细程度计算置信度
        confidence_factors = []
        
        # 检查直接影响环节数量
        segments = analysis_result.get("directly_affected_segments", [])
        if segments:
            confidence_factors.append(min(len(segments) * 0.1, 0.3))
        
        # 检查因果机制的详细程度
        mechanisms = analysis_result.get("causal_mechanisms", [])
        if mechanisms:
            confidence_factors.append(min(len(mechanisms) * 0.1, 0.3))
        
        # 检查证据强度
        strong_evidence = sum(1 for m in mechanisms if m.get("evidence_strength") == "strong")
        if strong_evidence > 0:
            confidence_factors.append(strong_evidence * 0.1)
        
        return min(sum(confidence_factors), 1.0) if confidence_factors else 0.5
    
    def _assess_transmission_complexity(self, analysis_result: Dict[str, Any]) -> str:
        """评估传导复杂性 - Assess transmission complexity"""
        total_effects = (
            len(analysis_result.get("second_order_effects", [])) +
            len(analysis_result.get("third_order_effects", [])) +
            len(analysis_result.get("spillover_effects", []))
        )
        
        if total_effects > 10:
            return "high"
        elif total_effects > 5:
            return "medium"
        else:
            return "low"
    
    def _map_confidence_to_score(self, confidence_level: str) -> float:
        """将置信度等级映射为数值 - Map confidence level to numerical score"""
        mapping = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4,
            "very_low": 0.2
        }
        return mapping.get(confidence_level.lower(), 0.5)
    
    def _determine_time_window(self, time_delay: str) -> str:
        """确定时间窗口 - Determine time window"""
        delay_mapping = {
            "immediate": "immediate",
            "short": "short_term",
            "medium": "medium_term",
            "long": "long_term"
        }
        
        for key, value in delay_mapping.items():
            if key in time_delay.lower():
                return value
        
        return "medium_term"  # 默认中期
    
    def _categorize_strength(self, strength_score: float) -> str:
        """分类强度等级 - Categorize strength level"""
        if strength_score > 0.7:
            return "strong"
        elif strength_score > 0.4:
            return "medium"
        else:
            return "weak"
    
    def _identify_temporal_pattern(self, transmission_paths: List[Dict[str, Any]]) -> str:
        """识别时间模式 - Identify temporal pattern"""
        immediate_paths = len([p for p in transmission_paths if p.get("time_window") == "immediate"])
        short_paths = len([p for p in transmission_paths if p.get("time_window") == "short_term"])
        long_paths = len([p for p in transmission_paths if p.get("time_window") in ["medium_term", "long_term"]])
        
        total_paths = len(transmission_paths)
        if total_paths == 0:
            return "unknown"
        
        immediate_ratio = immediate_paths / total_paths
        
        if immediate_ratio > 0.6:
            return "front_loaded"  # 前期密集
        elif long_paths > short_paths and long_paths > immediate_paths:
            return "delayed_impact"  # 延迟影响
        else:
            return "gradual_buildup"  # 逐步累积
    
    def _validate_logic_consistency(self, causal_graph: Dict[str, Any]) -> float:
        """验证逻辑一致性 - Validate logic consistency"""
        # 简化的逻辑一致性检查
        paths = causal_graph.get("transmission_paths", [])
        
        if not paths:
            return 0.0
        
        # 检查路径是否有逻辑冲突
        positive_paths = len([p for p in paths if p.get("overall_impact") == "positive"])
        negative_paths = len([p for p in paths if p.get("overall_impact") == "negative"])
        
        # 如果正面和负面路径比例过于极端，可能存在逻辑问题
        total_directional = positive_paths + negative_paths
        if total_directional > 0:
            balance_score = 1.0 - abs(positive_paths - negative_paths) / total_directional
            return min(balance_score + 0.5, 1.0)
        
        return 0.7  # 默认中等一致性
    
    def _validate_strength_reasonableness(self, causal_graph: Dict[str, Any]) -> float:
        """验证强度合理性 - Validate strength reasonableness"""
        strength_info = causal_graph.get("causal_strength", {})
        overall_strength = strength_info.get("overall_strength", 0.0)
        
        # 强度应该在合理范围内
        if 0.2 <= overall_strength <= 0.9:
            return 0.8
        elif 0.1 <= overall_strength <= 1.0:
            return 0.6
        else:
            return 0.3
    
    def _validate_temporal_consistency(self, causal_graph: Dict[str, Any]) -> float:
        """验证时间一致性 - Validate temporal consistency"""
        temporal_analysis = causal_graph.get("temporal_analysis", {})
        time_windows = temporal_analysis.get("time_windows", {})
        
        if not time_windows:
            return 0.5
        
        # 检查时间分布是否合理
        window_counts = [info.get("path_count", 0) for info in time_windows.values()]
        if max(window_counts) > 0:
            # 避免过度集中在单一时间窗口
            concentration = max(window_counts) / sum(window_counts)
            return 1.0 - concentration * 0.5
        
        return 0.7
    
    def _validate_assumption_reasonableness(self, causal_graph: Dict[str, Any]) -> float:
        """验证假设合理性 - Validate assumption reasonableness"""
        assumptions = causal_graph.get("key_assumptions", [])
        
        if not assumptions:
            return 0.5
        
        # 检查高重要性假设的比例
        high_importance = len([a for a in assumptions if a.get("importance") == "high"])
        total_assumptions = len(assumptions)
        
        if total_assumptions > 0:
            # 高重要性假设不应过多
            high_ratio = high_importance / total_assumptions
            if high_ratio < 0.3:
                return 0.8
            elif high_ratio < 0.6:
                return 0.6
            else:
                return 0.4
        
        return 0.6
    
    def _generate_improvement_recommendations(self, validation_details: Dict[str, Any]) -> List[str]:
        """生成改进建议 - Generate improvement recommendations"""
        recommendations = []
        
        if validation_details.get("logic_consistency", 1.0) < 0.6:
            recommendations.append("建议重新审视因果逻辑，确保正负面影响的平衡性")
        
        if validation_details.get("strength_reasonableness", 1.0) < 0.6:
            recommendations.append("建议调整因果强度评估，确保在合理范围内")
        
        if validation_details.get("temporal_consistency", 1.0) < 0.6:
            recommendations.append("建议重新分析时间动态，避免过度集中在单一时间窗口")
        
        if validation_details.get("assumption_reasonableness", 1.0) < 0.6:
            recommendations.append("建议减少高重要性假设的数量，增强推理稳健性")
        
        return recommendations
    
    # 后备方案方法 - Fallback methods
    def _fallback_direct_causality(self) -> Dict[str, Any]:
        """直接因果分析失败时的后备方案"""
        return {
            "directly_affected_segments": [],
            "affected_company_types": [],
            "causal_mechanisms": [],
            "analysis_confidence": 0.3
        }
    
    def _fallback_indirect_causality(self) -> Dict[str, Any]:
        """间接因果分析失败时的后备方案"""
        return {
            "second_order_effects": [],
            "third_order_effects": [],
            "spillover_effects": [],
            "feedback_loops": [],
            "transmission_complexity": "unknown"
        }
    
    def _fallback_key_assumptions(self) -> List[Dict[str, Any]]:
        """关键假设识别失败时的后备方案"""
        return [
            {
                "assumption_type": "市场假设",
                "assumption_content": "市场按正常逻辑反应",
                "importance": "medium",
                "validity_risk": "中等",
                "impact_if_violated": "分析准确性下降"
            }
        ]
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """
        获取推理统计信息
        Get reasoning statistics
        
        Returns:
            Dict[str, Any]: 推理统计信息 - Reasoning statistics
        """
        return {
            "engine_status": "active",
            "cached_reasoning_count": len(self.reasoning_cache),
            "supported_causal_types": list(self.causal_types.keys()),
            "strength_levels": list(self.reasoning_strength.keys()),
            "time_windows": list(self.time_windows.keys()),
            "llm_client_stats": self.llm_client.get_statistics()
        }

if __name__ == "__main__":
    # 测试因果推理引擎 - Test causal reasoning engine
    from utils.api_client import create_default_client
    
    print("测试因果推理引擎...")
    
    # 创建LLM客户端和因果推理引擎 - Create LLM client and causal reasoning engine
    llm_client = create_default_client()
    engine = CausalReasoningEngine(llm_client)
    
    # 模拟事件信息 - Mock event information
    test_event = {
        "basic_info": {
            "event_description": "新能源汽车补贴政策调整"
        },
        "event_classification": {
            "event_type": "policy",
            "event_type_cn": "政策类",
            "impact_direction": "mixed"
        },
        "entities": {
            "companies": ["比亚迪", "宁德时代", "特斯拉"]
        }
    }
    
    # 模拟产业链数据 - Mock industry chain data
    test_industry_chain = {
        "industry_name": "新能源汽车",
        "chain_structure": {
            "upstream": [{"segment": "锂电池材料", "importance": "high"}],
            "midstream": [{"segment": "整车制造", "importance": "high"}],
            "downstream": [{"segment": "销售服务", "importance": "medium"}]
        },
        "relationships": {
            "supply_relationships": [
                {
                    "from": "锂电池材料",
                    "to": "整车制造",
                    "strength": "strong"
                }
            ]
        }
    }
    
    # 创建因果图 - Create causal graph
    try:
        causal_graph = engine.create_causal_graph(test_event, test_industry_chain)
        
        if "error" not in causal_graph:
            print("✓ 因果图创建成功")
            print(f"传导路径数量: {causal_graph.get('reasoning_metadata', {}).get('total_paths', 0)}")
            print(f"高置信度路径: {causal_graph.get('reasoning_metadata', {}).get('high_confidence_paths', 0)}")
            
            # 验证推理链 - Validate reasoning chain
            validation = engine.validate_reasoning_chain(causal_graph)
            print(f"推理链验证: {'通过' if validation.get('is_valid') else '未通过'}")
            print(f"验证分数: {validation.get('validation_score', 0):.2f}")
            
        else:
            print(f"✗ 因果图创建失败: {causal_graph.get('error_details', 'N/A')}")
        
    except Exception as e:
        print(f"✗ 测试异常: {e}")
    
    print("测试完成")
