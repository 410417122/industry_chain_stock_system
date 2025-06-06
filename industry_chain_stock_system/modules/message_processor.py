"""
消息处理模块
Message Processing Module

负责解析财经消息，提取关键事件信息，并进行初步分析
Responsible for parsing financial messages, extracting key event information, and conducting preliminary analysis
"""

import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# 导入工具模块 - Import utility modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import LoggerMixin, log_execution_time
from utils.api_client import LLMClient

class MessageProcessor(LoggerMixin):
    """
    消息处理器类
    Message Processor Class
    
    负责财经消息的解析、分类和关键信息提取
    Responsible for parsing, categorizing and extracting key information from financial messages
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        初始化消息处理器
        Initialize message processor
        
        Args:
            llm_client: LLM客户端实例 - LLM client instance
        """
        self.llm_client = llm_client
        
        # 事件类型映射 - Event type mapping
        self.event_types = {
            "policy": "政策类",
            "technology": "技术类", 
            "market": "市场类",
            "emergency": "突发类",
            "financial": "财务类",
            "regulatory": "监管类"
        }
        
        # 影响程度映射 - Impact level mapping
        self.impact_levels = {
            "high": "高",
            "medium": "中", 
            "low": "低"
        }
        
        # 时效性映射 - Time effect mapping
        self.time_effects = {
            "immediate": "即时",
            "short_term": "短期",
            "medium_term": "中期", 
            "long_term": "长期"
        }
        
        self.logger.info("消息处理器初始化完成")
    
    @log_execution_time
    def parse_message(self, message_text: str) -> Dict[str, Any]:
        """
        解析财经消息，提取关键信息
        Parse financial message and extract key information
        
        Args:
            message_text: 财经消息文本 - Financial message text
            
        Returns:
            Dict[str, Any]: 解析后的结构化信息 - Parsed structured information
        """
        self.logger.info(f"开始解析消息，长度: {len(message_text)}字符")
        
        try:
            # 第1步：基础信息提取 - Step 1: Basic information extraction
            basic_info = self._extract_basic_info(message_text)
            
            # 第2步：事件分类 - Step 2: Event classification
            event_classification = self._classify_event(message_text)
            
            # 第3步：影响分析 - Step 3: Impact analysis
            impact_analysis = self._analyze_impact(message_text, event_classification)
            
            # 第4步：关键实体识别 - Step 4: Key entity recognition
            entities = self._extract_entities(message_text)
            
            # 第5步：情感分析 - Step 5: Sentiment analysis
            sentiment = self._analyze_sentiment(message_text)
            
            # 整合所有信息 - Integrate all information
            parsed_result = {
                "original_message": message_text,
                "message_length": len(message_text),
                "parse_timestamp": datetime.now().isoformat(),
                "basic_info": basic_info,
                "event_classification": event_classification,
                "impact_analysis": impact_analysis,
                "entities": entities,
                "sentiment": sentiment,
                "confidence_score": self._calculate_confidence(
                    basic_info, event_classification, impact_analysis
                )
            }
            
            self.logger.info("消息解析完成")
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"消息解析失败: {e}")
            return {
                "error": "消息解析失败",
                "error_details": str(e),
                "original_message": message_text,
                "parse_timestamp": datetime.now().isoformat()
            }
    
    def _extract_basic_info(self, message_text: str) -> Dict[str, Any]:
        """
        提取消息基础信息
        Extract basic information from message
        
        Args:
            message_text: 消息文本 - Message text
            
        Returns:
            Dict[str, Any]: 基础信息 - Basic information
        """
        # 构建基础信息提取提示 - Build basic info extraction prompt
        prompt = f"""
        请分析以下财经消息，提取基础信息：

        消息内容：{message_text}

        请提取以下信息：
        1. 消息主题（简洁概括）
        2. 涉及的主要主体（公司、机构、行业等）
        3. 关键事件描述（一句话概括）
        4. 消息发布时间线索（如果有）
        5. 消息来源线索（如果有）

        请以JSON格式返回，包含以下字段：
        {{
            "subject": "消息主题",
            "main_entities": ["主体1", "主体2"],
            "event_description": "关键事件描述",
            "time_clues": "时间线索",
            "source_clues": "来源线索"
        }}
        """
        
        try:
            response = self.llm_client.generate_structured(prompt)
            
            # 验证和清理响应 - Validate and clean response
            if "error" not in response:
                return {
                    "subject": response.get("subject", "未知主题"),
                    "main_entities": response.get("main_entities", []),
                    "event_description": response.get("event_description", "未知事件"),
                    "time_clues": response.get("time_clues", "无时间线索"),
                    "source_clues": response.get("source_clues", "无来源线索")
                }
            else:
                self.logger.warning(f"基础信息提取失败: {response}")
                return self._fallback_basic_info(message_text)
                
        except Exception as e:
            self.logger.error(f"基础信息提取异常: {e}")
            return self._fallback_basic_info(message_text)
    
    def _classify_event(self, message_text: str) -> Dict[str, Any]:
        """
        对事件进行分类
        Classify the event
        
        Args:
            message_text: 消息文本 - Message text
            
        Returns:
            Dict[str, Any]: 事件分类结果 - Event classification result
        """
        # 构建事件分类提示 - Build event classification prompt
        prompt = f"""
        请对以下财经消息进行事件分类：

        消息内容：{message_text}

        请从以下维度进行分类：

        1. 事件类型：
        - policy: 政策类（政府政策、法规变化等）
        - technology: 技术类（技术突破、产品发布等）
        - market: 市场类（市场变化、供需关系等）
        - emergency: 突发类（突发事件、危机等）
        - financial: 财务类（财报、业绩、融资等）
        - regulatory: 监管类（监管变化、合规要求等）

        2. 影响范围：
        - individual: 个股级别
        - industry: 行业级别
        - sector: 板块级别
        - market: 全市场级别

        3. 影响方向：
        - positive: 正面影响
        - negative: 负面影响
        - neutral: 中性影响
        - mixed: 混合影响

        4. 时效性：
        - immediate: 即时影响
        - short_term: 短期影响（1-3个月）
        - medium_term: 中期影响（3-12个月）
        - long_term: 长期影响（1年以上）

        请以JSON格式返回：
        {{
            "event_type": "事件类型",
            "impact_scope": "影响范围",
            "impact_direction": "影响方向", 
            "time_effect": "时效性",
            "classification_reason": "分类理由"
        }}
        """
        
        try:
            response = self.llm_client.generate_structured(prompt)
            
            # 验证分类结果 - Validate classification result
            if "error" not in response:
                return {
                    "event_type": response.get("event_type", "unknown"),
                    "event_type_cn": self.event_types.get(response.get("event_type", "unknown"), "未知类型"),
                    "impact_scope": response.get("impact_scope", "unknown"),
                    "impact_direction": response.get("impact_direction", "neutral"),
                    "time_effect": response.get("time_effect", "unknown"),
                    "time_effect_cn": self.time_effects.get(response.get("time_effect", "unknown"), "未知时效"),
                    "classification_reason": response.get("classification_reason", "无分类理由")
                }
            else:
                return self._fallback_classification()
                
        except Exception as e:
            self.logger.error(f"事件分类异常: {e}")
            return self._fallback_classification()
    
    def _analyze_impact(self, message_text: str, event_classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析事件影响
        Analyze event impact
        
        Args:
            message_text: 消息文本 - Message text
            event_classification: 事件分类结果 - Event classification result
            
        Returns:
            Dict[str, Any]: 影响分析结果 - Impact analysis result
        """
        # 构建影响分析提示 - Build impact analysis prompt
        prompt = f"""
        基于以下财经消息和事件分类，分析其可能的影响：

        消息内容：{message_text}

        事件分类：
        - 事件类型：{event_classification.get('event_type_cn', '未知')}
        - 影响范围：{event_classification.get('impact_scope', '未知')}
        - 影响方向：{event_classification.get('impact_direction', '未知')}

        请分析：
        1. 影响强度（高/中/低）
        2. 受影响的主要行业（列出3-5个）
        3. 可能受益的股票类型
        4. 可能受损的股票类型
        5. 影响的主要传导机制
        6. 不确定性因素

        请以JSON格式返回：
        {{
            "impact_intensity": "影响强度",
            "affected_industries": ["行业1", "行业2"],
            "beneficiary_types": ["受益类型1", "受益类型2"],
            "affected_types": ["受损类型1", "受损类型2"],
            "transmission_mechanism": "传导机制描述",
            "uncertainty_factors": ["不确定因素1", "不确定因素2"]
        }}
        """
        
        try:
            response = self.llm_client.generate_structured(prompt)
            
            if "error" not in response:
                return {
                    "impact_intensity": response.get("impact_intensity", "中"),
                    "impact_intensity_en": self._map_intensity_to_en(response.get("impact_intensity", "中")),
                    "affected_industries": response.get("affected_industries", []),
                    "beneficiary_types": response.get("beneficiary_types", []),
                    "affected_types": response.get("affected_types", []),
                    "transmission_mechanism": response.get("transmission_mechanism", "未知传导机制"),
                    "uncertainty_factors": response.get("uncertainty_factors", [])
                }
            else:
                return self._fallback_impact_analysis()
                
        except Exception as e:
            self.logger.error(f"影响分析异常: {e}")
            return self._fallback_impact_analysis()
    
    def _extract_entities(self, message_text: str) -> Dict[str, Any]:
        """
        提取关键实体
        Extract key entities
        
        Args:
            message_text: 消息文本 - Message text
            
        Returns:
            Dict[str, Any]: 实体提取结果 - Entity extraction result
        """
        # 构建实体提取提示 - Build entity extraction prompt  
        prompt = f"""
        请从以下财经消息中提取关键实体：

        消息内容：{message_text}

        请提取以下类型的实体：
        1. 公司名称（包括简称和全称）
        2. 股票代码（如果有）
        3. 行业名称
        4. 产品或服务名称
        5. 地理位置
        6. 人名（高管、分析师等）
        7. 机构名称（政府部门、监管机构等）
        8. 数值信息（金额、百分比、日期等）

        请以JSON格式返回：
        {{
            "companies": ["公司1", "公司2"],
            "stock_codes": ["代码1", "代码2"],
            "industries": ["行业1", "行业2"],
            "products_services": ["产品1", "服务1"],
            "locations": ["地点1", "地点2"],
            "people": ["人名1", "人名2"],
            "institutions": ["机构1", "机构2"],
            "numerical_info": ["数值1", "数值2"]
        }}
        """
        
        try:
            response = self.llm_client.generate_structured(prompt)
            
            if "error" not in response:
                # 清理和验证实体 - Clean and validate entities
                cleaned_entities = {}
                for key, values in response.items():
                    if isinstance(values, list):
                        # 过滤空值和重复项 - Filter empty values and duplicates
                        cleaned_values = list(set([v.strip() for v in values if v and v.strip()]))
                        cleaned_entities[key] = cleaned_values[:10]  # 限制数量 - Limit quantity
                    else:
                        cleaned_entities[key] = []
                
                return cleaned_entities
            else:
                return self._fallback_entities()
                
        except Exception as e:
            self.logger.error(f"实体提取异常: {e}")
            return self._fallback_entities()
    
    def _analyze_sentiment(self, message_text: str) -> Dict[str, Any]:
        """
        分析消息情感倾向
        Analyze message sentiment
        
        Args:
            message_text: 消息文本 - Message text
            
        Returns:
            Dict[str, Any]: 情感分析结果 - Sentiment analysis result
        """
        # 构建情感分析提示 - Build sentiment analysis prompt
        prompt = f"""
        请分析以下财经消息的情感倾向：

        消息内容：{message_text}

        请从以下维度进行分析：
        1. 整体情感倾向（正面/负面/中性）
        2. 情感强度（强/中/弱）
        3. 主要情感关键词
        4. 情感分析理由

        请以JSON格式返回：
        {{
            "sentiment": "情感倾向",
            "intensity": "情感强度",
            "keywords": ["关键词1", "关键词2"],
            "reason": "分析理由"
        }}
        """
        
        try:
            response = self.llm_client.generate_structured(prompt)
            
            if "error" not in response:
                sentiment = response.get("sentiment", "中性")
                return {
                    "sentiment": sentiment,
                    "sentiment_score": self._map_sentiment_to_score(sentiment),
                    "intensity": response.get("intensity", "中"),
                    "keywords": response.get("keywords", []),
                    "reason": response.get("reason", "无分析理由")
                }
            else:
                return self._fallback_sentiment()
                
        except Exception as e:
            self.logger.error(f"情感分析异常: {e}")
            return self._fallback_sentiment()
    
    def _calculate_confidence(
        self, 
        basic_info: Dict[str, Any], 
        event_classification: Dict[str, Any], 
        impact_analysis: Dict[str, Any]
    ) -> float:
        """
        计算解析结果的置信度
        Calculate confidence score of parsing result
        
        Args:
            basic_info: 基础信息 - Basic information
            event_classification: 事件分类 - Event classification
            impact_analysis: 影响分析 - Impact analysis
            
        Returns:
            float: 置信度分数（0-1） - Confidence score (0-1)
        """
        confidence_score = 0.0
        
        # 基础信息完整性检查 - Basic info completeness check
        if basic_info.get("subject") and basic_info.get("subject") != "未知主题":
            confidence_score += 0.2
        if basic_info.get("main_entities") and len(basic_info["main_entities"]) > 0:
            confidence_score += 0.15
        if basic_info.get("event_description") and basic_info.get("event_description") != "未知事件":
            confidence_score += 0.15
        
        # 事件分类准确性检查 - Event classification accuracy check
        if event_classification.get("event_type") and event_classification.get("event_type") != "unknown":
            confidence_score += 0.2
        if event_classification.get("impact_direction") and event_classification.get("impact_direction") != "neutral":
            confidence_score += 0.1
        
        # 影响分析详细程度检查 - Impact analysis detail check
        if impact_analysis.get("affected_industries") and len(impact_analysis["affected_industries"]) > 0:
            confidence_score += 0.1
        if impact_analysis.get("transmission_mechanism") and impact_analysis.get("transmission_mechanism") != "未知传导机制":
            confidence_score += 0.1
        
        return min(confidence_score, 1.0)  # 确保不超过1.0 - Ensure not exceeding 1.0
    
    # 辅助方法 - Helper methods
    def _fallback_basic_info(self, message_text: str) -> Dict[str, Any]:
        """基础信息提取失败时的后备方案 - Fallback for basic info extraction failure"""
        return {
            "subject": "信息提取失败",
            "main_entities": [],
            "event_description": f"消息长度{len(message_text)}字符",
            "time_clues": "无",
            "source_clues": "无"
        }
    
    def _fallback_classification(self) -> Dict[str, Any]:
        """事件分类失败时的后备方案 - Fallback for event classification failure"""
        return {
            "event_type": "unknown",
            "event_type_cn": "未知类型",
            "impact_scope": "unknown",
            "impact_direction": "neutral",
            "time_effect": "unknown",
            "time_effect_cn": "未知时效",
            "classification_reason": "分类失败"
        }
    
    def _fallback_impact_analysis(self) -> Dict[str, Any]:
        """影响分析失败时的后备方案 - Fallback for impact analysis failure"""
        return {
            "impact_intensity": "低",
            "impact_intensity_en": "low",
            "affected_industries": [],
            "beneficiary_types": [],
            "affected_types": [],
            "transmission_mechanism": "分析失败",
            "uncertainty_factors": ["分析不确定"]
        }
    
    def _fallback_entities(self) -> Dict[str, Any]:
        """实体提取失败时的后备方案 - Fallback for entity extraction failure"""
        return {
            "companies": [],
            "stock_codes": [],
            "industries": [],
            "products_services": [],
            "locations": [],
            "people": [],
            "institutions": [],
            "numerical_info": []
        }
    
    def _fallback_sentiment(self) -> Dict[str, Any]:
        """情感分析失败时的后备方案 - Fallback for sentiment analysis failure"""
        return {
            "sentiment": "中性",
            "sentiment_score": 0.0,
            "intensity": "弱",
            "keywords": [],
            "reason": "分析失败"
        }
    
    def _map_intensity_to_en(self, intensity_cn: str) -> str:
        """将中文影响强度映射为英文 - Map Chinese intensity to English"""
        mapping = {"高": "high", "中": "medium", "低": "low"}
        return mapping.get(intensity_cn, "unknown")
    
    def _map_sentiment_to_score(self, sentiment: str) -> float:
        """将情感倾向映射为数值分数 - Map sentiment to numerical score"""
        mapping = {
            "正面": 0.7, "积极": 0.7, "利好": 0.8,
            "负面": -0.7, "消极": -0.7, "利空": -0.8,
            "中性": 0.0, "中立": 0.0
        }
        return mapping.get(sentiment, 0.0)
    
    def batch_parse_messages(self, messages: List[str]) -> List[Dict[str, Any]]:
        """
        批量解析消息
        Batch parse messages
        
        Args:
            messages: 消息列表 - List of messages
            
        Returns:
            List[Dict[str, Any]]: 解析结果列表 - List of parsing results
        """
        self.logger.info(f"开始批量解析 {len(messages)} 条消息")
        
        results = []
        for i, message in enumerate(messages):
            self.logger.debug(f"解析第 {i+1}/{len(messages)} 条消息")
            try:
                result = self.parse_message(message)
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                self.logger.error(f"第 {i+1} 条消息解析失败: {e}")
                results.append({
                    "batch_index": i,
                    "error": "解析失败",
                    "error_details": str(e),
                    "original_message": message
                })
        
        self.logger.info(f"批量解析完成，成功 {len([r for r in results if 'error' not in r])} 条，失败 {len([r for r in results if 'error' in r])} 条")
        return results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        获取处理统计信息
        Get processing statistics
        
        Returns:
            Dict[str, Any]: 统计信息 - Statistics
        """
        # 这里可以添加处理统计逻辑 - Processing statistics logic can be added here
        # 目前返回基础信息 - Currently returning basic information
        return {
            "processor_status": "active",
            "llm_client_stats": self.llm_client.get_statistics(),
            "supported_event_types": list(self.event_types.keys()),
            "supported_impact_levels": list(self.impact_levels.keys()),
            "supported_time_effects": list(self.time_effects.keys())
        }

if __name__ == "__main__":
    # 测试消息处理器 - Test message processor
    from utils.api_client import create_default_client
    
    print("测试消息处理器...")
    
    # 创建LLM客户端和消息处理器 - Create LLM client and message processor
    llm_client = create_default_client()
    processor = MessageProcessor(llm_client)
    
    # 测试消息 - Test message
    test_message = """
    财政部等四部门发布通知，新能源汽车推广应用财政补贴政策年底到期后不再延续，
    转为支持充电基础设施建设和配套运营服务等方面。此次政策调整将对新能源汽车
    产业链产生重要影响，预计充电桩制造商将受益，而传统车企需要加快转型步伐。
    """
    
    # 解析消息 - Parse message
    try:
        result = processor.parse_message(test_message)
        
        print("✓ 消息解析成功")
        print(f"消息主题: {result.get('basic_info', {}).get('subject', 'N/A')}")
        print(f"事件类型: {result.get('event_classification', {}).get('event_type_cn', 'N/A')}")
        print(f"影响方向: {result.get('event_classification', {}).get('impact_direction', 'N/A')}")
        print(f"置信度: {result.get('confidence_score', 0):.2f}")
        
        # 显示统计信息 - Show statistics
        stats = processor.get_processing_statistics()
        print(f"处理器状态: {stats['processor_status']}")
        
    except Exception as e:
        print(f"✗ 消息解析失败: {e}")
    
    print("测试完成")
