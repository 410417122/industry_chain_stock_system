"""
投资决策模块
Investment Decision Module

负责基于因果分析结果生成投资建议、进行风险评估和生成报告
Responsible for generating investment advice, conducting risk assessment, and creating reports based on causal analysis results
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

class InvestmentAdvisor(LoggerMixin):
    """
    投资顾问类
    Investment Advisor Class
    
    负责基于分析结果生成投资建议和报告
    Responsible for generating investment advice and reports based on analysis results
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        初始化投资顾问
        Initialize investment advisor
        
        Args:
            llm_client: LLM客户端实例 - LLM client instance
        """
        self.llm_client = llm_client
        
        # 投资建议类型 - Investment advice types
        self.advice_types = {
            "buy": "买入",
            "sell": "卖出",
            "hold": "持有",
            "watch": "关注"
        }
        
        # 风险等级 - Risk levels
        self.risk_levels = {
            "low": "低风险",
            "medium": "中风险",
            "high": "高风险"
        }
        
        # 投资时间窗口 - Investment time windows
        self.investment_horizons = {
            "short_term": "短期（1-3个月）",
            "medium_term": "中期（3-12个月）",
            "long_term": "长期（1年以上）"
        }
        
        self.logger.info("投资顾问初始化完成")
    
    @log_execution_time
    def generate_recommendations(
        self, 
        causal_graph: Dict[str, Any], 
        industry_chain: Dict[str, Any],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        生成投资建议
        Generate investment recommendations
        
        Args:
            causal_graph: 因果图数据 - Causal graph data
            industry_chain: 产业链数据 - Industry chain data
            top_n: 推荐股票数量 - Number of recommended stocks
            
        Returns:
            List[Dict[str, Any]]: 投资建议列表 - List of investment recommendations
        """
        self.logger.info("开始生成投资建议")
        
        try:
            # 第1步：提取关键影响信息 - Step 1: Extract key impact information
            key_impacts = self._extract_key_impacts(causal_graph)
            
            # 第2步：筛选相关股票 - Step 2: Filter relevant stocks
            relevant_stocks = self._filter_relevant_stocks(key_impacts, industry_chain)
            
            # 第3步：评估股票潜力 - Step 3: Evaluate stock potential
            stock_potentials = self._evaluate_stock_potential(relevant_stocks, key_impacts, causal_graph)
            
            # 第4步：生成投资建议 - Step 4: Generate investment advice
            recommendations = self._formulate_investment_advice(stock_potentials, top_n)
            
            # 第5步：进行风险评估 - Step 5: Conduct risk assessment
            recommendations_with_risk = self._assess_recommendation_risk(recommendations, causal_graph)
            
            self.logger.info(f"投资建议生成完成，共推荐 {len(recommendations_with_risk)} 只股票")
            return recommendations_with_risk
            
        except Exception as e:
            self.logger.error(f"投资建议生成失败: {e}")
            return []
    
    def _extract_key_impacts(self, causal_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取关键影响信息
        Extract key impact information
        
        Args:
            causal_graph: 因果图数据 - Causal graph data
            
        Returns:
            Dict[str, Any]: 关键影响信息 - Key impact information
        """
        # 提取直接影响和主要传导路径 - Extract direct impacts and main transmission paths
        direct_impacts = causal_graph.get("direct_causality", {}).get("directly_affected_segments", [])
        transmission_paths = causal_graph.get("transmission_paths", [])
        
        # 筛选高置信度和强影响的路径 - Filter high-confidence and strong-impact paths
        strong_paths = [
            p for p in transmission_paths 
            if p.get("confidence", 0) > 0.6 and 
               causal_graph.get("causal_strength", {}).get("path_strengths", {}).get(p.get("path_id"), {}).get("strength_score", 0) > 0.6
        ]
        
        # 确定主要影响方向和强度 - Determine main impact direction and intensity
        overall_impact_direction = causal_graph.get("event_info", {}).get("event_classification", {}).get("impact_direction", "mixed")
        overall_impact_intensity = causal_graph.get("causal_strength", {}).get("overall_strength", 0.5)
        
        return {
            "direct_impacts": direct_impacts,
            "strong_transmission_paths": strong_paths,
            "overall_impact_direction": overall_impact_direction,
            "overall_impact_intensity": overall_impact_intensity,
            "affected_company_types": causal_graph.get("direct_causality", {}).get("affected_company_types", [])
        }
    
    def _filter_relevant_stocks(
        self, 
        key_impacts: Dict[str, Any], 
        industry_chain: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        筛选相关股票
        Filter relevant stocks
        
        Args:
            key_impacts: 关键影响信息 - Key impact information
            industry_chain: 产业链数据 - Industry chain data
            
        Returns:
            List[Dict[str, Any]]: 相关股票列表 - List of relevant stocks
        """
        # 获取行业内所有公司 - Get all companies in the industry
        all_companies = industry_chain.get("companies", [])
        
        # 筛选受影响类型的公司 - Filter companies of affected types
        affected_types = key_impacts.get("affected_company_types", [])
        relevant_stocks = [
            company for company in all_companies
            if company.get("market_status") in affected_types or company.get("position") in affected_types
        ]
        
        # 进一步基于产业链环节筛选 - Further filter based on industry chain segments
        affected_segments = [seg.get("segment_name") for seg in key_impacts.get("direct_impacts", [])]
        
        # 使用LLM辅助筛选 - Use LLM to assist filtering
        prompt = f"""
        基于以下受影响的产业链环节和企业类型，筛选出最相关的上市公司：

        受影响环节：{', '.join(affected_segments)}
        受影响企业类型：{', '.join(affected_types)}
        
        可选企业列表（部分）：
        {json.dumps([{"name": c.get("name"), "business": c.get("business")} for c in all_companies[:15]], ensure_ascii=False, indent=2)}

        请列出10-15家最相关的上市公司，并说明其与事件的相关性。
        以JSON格式返回：
        {{
            "relevant_stocks": [
                {{
                    "name": "公司名称",
                    "code": "股票代码",
                    "relevance_reason": "相关性理由"
                }}
            ]
        }}
        """
        
        try:
            response = self.llm_client.generate_structured(prompt)
            
            if "error" not in response and "relevant_stocks" in response:
                llm_filtered_stocks = response["relevant_stocks"]
                
                # 合并并去重 - Merge and deduplicate
                final_stocks = []
                seen_names = set()
                
                for stock_info in llm_filtered_stocks:
                    if isinstance(stock_info, dict) and "name" in stock_info:
                        # 查找原始公司数据 - Find original company data
                        original_company = next((c for c in all_companies if c.get("name") == stock_info.get("name")), None)
                        if original_company and original_company.get("name") not in seen_names:
                            final_stocks.append({**original_company, "relevance_reason": stock_info.get("relevance_reason")})
                            seen_names.add(original_company.get("name"))
                
                return final_stocks[:20]  # 限制数量
            else:
                return relevant_stocks[:20] # 使用初步筛选结果
                
        except Exception as e:
            self.logger.error(f"LLM股票筛选异常: {e}")
            return relevant_stocks[:20] # 使用初步筛选结果
    
    def _evaluate_stock_potential(
        self, 
        relevant_stocks: List[Dict[str, Any]], 
        key_impacts: Dict[str, Any],
        causal_graph: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        评估股票潜力
        Evaluate stock potential
        
        Args:
            relevant_stocks: 相关股票列表 - List of relevant stocks
            key_impacts: 关键影响信息 - Key impact information
            causal_graph: 因果图数据 - Causal graph data
            
        Returns:
            List[Dict[str, Any]]: 股票潜力评估列表 - List of stock potential evaluations
        """
        stock_potentials = []
        
        for stock in relevant_stocks:
            # 构建股票潜力评估提示 - Build stock potential evaluation prompt
            prompt = f"""
            请评估以下股票在当前事件影响下的投资潜力：

            股票信息：
            - 名称：{stock.get('name', '未知')}
            - 代码：{stock.get('code', '未知')}
            - 主要业务：{stock.get('business', '未知')}
            - 产业链位置：{stock.get('position', '未知')}

            事件关键影响：
            - 主要影响方向：{key_impacts.get('overall_impact_direction', '未知')}
            - 整体影响强度：{key_impacts.get('overall_impact_intensity', 0.5):.2f}
            - 主要影响环节：{', '.join([seg.get('segment_name') for seg in key_impacts.get('direct_impacts', [])])}

            因果分析摘要：
            - 强传导路径数量：{len(key_impacts.get('strong_transmission_paths', []))}
            - 总体不确定性：{causal_graph.get('uncertainty_analysis', {}).get('overall_uncertainty', 0.5):.2f}

            请评估：
            1. 潜在影响程度（高/中/低）
            2. 影响时间窗口（短期/中期/长期）
            3. 投资确定性（高/中/低）
            4. 主要看涨/看跌理由
            5. 潜在风险因素

            以JSON格式返回：
            {{
                "potential_impact": "影响程度",
                "time_horizon": "时间窗口",
                "certainty": "确定性",
                "bullish_reasons": ["看涨理由1", "看涨理由2"],
                "bearish_reasons": ["看跌理由1", "看跌理由2"],
                "risk_factors": ["风险因素1", "风险因素2"]
            }}
            """
            
            try:
                response = self.llm_client.generate_structured(prompt)
                
                if "error" not in response:
                    potential_info = {
                        "stock_code": stock.get("code"),
                        "stock_name": stock.get("name"),
                        "potential_impact": response.get("potential_impact", "中"),
                        "time_horizon": response.get("time_horizon", "中期"),
                        "certainty": response.get("certainty", "中"),
                        "bullish_reasons": response.get("bullish_reasons", []),
                        "bearish_reasons": response.get("bearish_reasons", []),
                        "risk_factors": response.get("risk_factors", [])
                    }
                    stock_potentials.append(potential_info)
                else:
                    self.logger.warning(f"股票潜力评估失败: {stock.get('name')}")
                    
            except Exception as e:
                self.logger.error(f"股票潜力评估异常: {e}")
        
        return stock_potentials
    
    def _formulate_investment_advice(
        self, 
        stock_potentials: List[Dict[str, Any]], 
        top_n: int
    ) -> List[Dict[str, Any]]:
        """
        制定投资建议
        Formulate investment advice
        
        Args:
            stock_potentials: 股票潜力评估 - Stock potential evaluations
            top_n: 推荐数量 - Number of recommendations
            
        Returns:
            List[Dict[str, Any]]: 投资建议列表 - List of investment recommendations
        """
        # 根据潜力排序 - Sort by potential
        # 简化排序逻辑：优先考虑高影响、高确定性 - Simplified sorting logic: prioritize high impact, high certainty
        def sort_key(stock):
            impact_score = {"高": 3, "中": 2, "低": 1}.get(stock.get("potential_impact", "低"), 1)
            certainty_score = {"高": 3, "中": 2, "低": 1}.get(stock.get("certainty", "低"), 1)
            return impact_score * 10 + certainty_score  # 组合评分
        
        sorted_stocks = sorted(stock_potentials, key=sort_key, reverse=True)
        
        # 选择Top N - Select Top N
        recommendations = []
        for stock in sorted_stocks[:top_n]:
            advice = {
                "stock_code": stock.get("stock_code"),
                "stock_name": stock.get("stock_name"),
                "advice_type": self._determine_advice_type(stock),
                "time_horizon": stock.get("time_horizon"),
                "time_horizon_cn": self.investment_horizons.get(self._map_time_horizon_to_key(stock.get("time_horizon")), "未知"),
                "rationale": ", ".join(stock.get("bullish_reasons", []) or stock.get("bearish_reasons", [])),
                "potential_impact": stock.get("potential_impact"),
                "certainty": stock.get("certainty"),
                "key_risks": stock.get("risk_factors", [])
            }
            recommendations.append(advice)
            
        return recommendations
    
    def _assess_recommendation_risk(
        self, 
        recommendations: List[Dict[str, Any]],
        causal_graph: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        评估建议风险
        Assess recommendation risk
        
        Args:
            recommendations: 投资建议 - Investment recommendations
            causal_graph: 因果图数据 - Causal graph data
            
        Returns:
            List[Dict[str, Any]]: 带风险评估的建议 - Recommendations with risk assessment
        """
        # 获取不确定性信息 - Get uncertainty information
        uncertainty_analysis = causal_graph.get("uncertainty_analysis", {})
        overall_uncertainty = uncertainty_analysis.get("overall_uncertainty", 0.5)
        
        for rec in recommendations:
            # 简化风险评估 - Simplified risk assessment
            stock_certainty = {"高": 0.8, "中": 0.6, "低": 0.4}.get(rec.get("certainty", "低"), 0.4)
            
            # 综合不确定性和股票确定性 - Combine overall uncertainty and stock certainty
            risk_score = (1 - stock_certainty) * 0.6 + overall_uncertainty * 0.4
            
            if risk_score > 0.7:
                risk_level = "high"
            elif risk_score > 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            rec["risk_level"] = risk_level
            rec["risk_level_cn"] = self.risk_levels.get(risk_level, "未知风险")
            rec["risk_assessment_details"] = {
                "overall_analysis_uncertainty": overall_uncertainty,
                "stock_specific_certainty": stock_certainty,
                "calculated_risk_score": risk_score
            }
            
        return recommendations
    
    def generate_report(
        self, 
        analysis_result: Dict[str, Any], 
        output_format: str = "text"
    ) -> Union[str, Dict[str, Any]]:
        """
        生成分析报告
        Generate analysis report
        
        Args:
            analysis_result: 分析结果 - Analysis result
            output_format: 输出格式 - Output format ("text", "json", "html")
            
        Returns:
            Union[str, Dict[str, Any]]: 生成的报告 - Generated report
        """
        self.logger.info(f"开始生成分析报告，格式: {output_format}")
        
        if output_format == "json":
            return analysis_result
        
        elif output_format == "text":
            return self._generate_text_report(analysis_result)
        
        elif output_format == "html":
            return self._generate_html_report(analysis_result) # 待实现
            
        else:
            self.logger.warning(f"不支持的报告格式: {output_format}，将返回JSON格式")
            return analysis_result
    
    def _generate_text_report(self, result: Dict[str, Any]) -> str:
        """生成文本格式报告 - Generate text format report"""
        report_parts = []
        
        # 报告头部 - Report header
        report_parts.append("="*50)
        report_parts.append("        消息驱动的产业链选股系统 - 分析报告")
        report_parts.append("="*50)
        report_parts.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_parts.append(f"分析模式: {result.get('analysis_mode', '未知')}")
        report_parts.append(f"输入消息: {result.get('input_message', '')[:100]}...")
        report_parts.append("-"*50)
        
        # 事件分析 - Event analysis
        event_info = result.get("event_info", {})
        report_parts.append("【事件分析】")
        report_parts.append(f"  主题: {event_info.get('basic_info', {}).get('subject', '未知')}")
        report_parts.append(f"  类型: {event_info.get('event_classification', {}).get('event_type_cn', '未知')}")
        report_parts.append(f"  影响方向: {event_info.get('event_classification', {}).get('impact_direction', '未知')}")
        report_parts.append(f"  置信度: {event_info.get('confidence_score', 0):.2f}")
        report_parts.append("-"*50)
        
        # 产业链影响 - Industry chain impact
        causal_analysis = result.get("causal_analysis", {})
        report_parts.append("【产业链影响分析】")
        report_parts.append(f"  影响行业: {causal_analysis.get('industry_chain_name', '未知')}")
        report_parts.append(f"  直接影响环节: {len(causal_analysis.get('direct_causality', {}).get('directly_affected_segments', []))}个")
        report_parts.append(f"  主要传导路径: {len(causal_analysis.get('transmission_paths', []))}条")
        report_parts.append(f"  整体因果强度: {causal_analysis.get('causal_strength', {}).get('overall_strength', 0):.2f}")
        report_parts.append("-"*50)
        
        # 投资建议 - Investment recommendations
        recommendations = result.get("investment_recommendations", [])
        report_parts.append("【投资建议】")
        if recommendations:
            for i, rec in enumerate(recommendations):
                report_parts.append(f"  {i+1}. {rec.get('stock_name', '未知')} ({rec.get('stock_code', 'N/A')})")
                report_parts.append(f"     建议操作: {self.advice_types.get(rec.get('advice_type'), '未知')}")
                report_parts.append(f"     时间窗口: {rec.get('time_horizon_cn', '未知')}")
                report_parts.append(f"     核心逻辑: {rec.get('rationale', '无')[:80]}...")
                report_parts.append(f"     风险等级: {rec.get('risk_level_cn', '未知')}")
        else:
            report_parts.append("  暂无明确投资建议")
        report_parts.append("="*50)
        
        return "\n".join(report_parts)
    
    def _generate_html_report(self, result: Dict[str, Any]) -> str:
        """生成HTML格式报告（待实现）- Generate HTML format report (to be implemented)"""
        # 此处应使用模板引擎（如Jinja2）生成HTML报告
        # This part should use a template engine (e.g., Jinja2) to generate HTML report
        self.logger.warning("HTML报告生成功能尚未完全实现，返回文本格式")
        return self._generate_text_report(result)
    
    # 辅助方法 - Helper methods
    def _determine_advice_type(self, stock_potential: Dict[str, Any]) -> str:
        """确定建议类型 - Determine advice type"""
        impact = stock_potential.get("potential_impact", "中")
        certainty = stock_potential.get("certainty", "中")
        
        if impact == "高" and certainty == "高":
            return "buy"
        elif impact == "高" and certainty == "中":
            return "buy"
        elif impact == "中" and certainty == "高":
            return "watch"
        elif impact == "负面" and certainty == "高": # 假设影响方向已整合
            return "sell"
        else:
            return "hold" # 默认持有或观望
    
    def _map_time_horizon_to_key(self, horizon_cn: str) -> str:
        """将中文时间窗口映射回英文键 - Map Chinese time horizon back to English key"""
        for key, value in self.investment_horizons.items():
            if value == horizon_cn:
                return key
        return "medium_term" # 默认
    
    def get_advisor_statistics(self) -> Dict[str, Any]:
        """
        获取投资顾问统计信息
        Get investment advisor statistics
        
        Returns:
            Dict[str, Any]: 统计信息 - Statistics
        """
        return {
            "advisor_status": "active",
            "supported_advice_types": list(self.advice_types.keys()),
            "supported_risk_levels": list(self.risk_levels.keys()),
            "supported_horizons": list(self.investment_horizons.keys()),
            "llm_client_stats": self.llm_client.get_statistics()
        }

if __name__ == "__main__":
    # 测试投资顾问 - Test investment advisor
    from utils.api_client import create_default_client
    
    print("测试投资顾问...")
    
    # 创建LLM客户端和投资顾问 - Create LLM client and investment advisor
    llm_client = create_default_client()
    advisor = InvestmentAdvisor(llm_client)
    
    # 模拟因果图和产业链数据 - Mock causal graph and industry chain data
    test_causal_graph = {
        "event_info": {"event_classification": {"impact_direction": "positive"}},
        "direct_causality": {
            "directly_affected_segments": [{"segment_name": "充电桩制造"}],
            "affected_company_types": ["充电桩制造商"]
        },
        "transmission_paths": [{"path_id": "p1", "confidence": 0.8, "path_length": 1}],
        "causal_strength": {"overall_strength": 0.75, "path_strengths": {"p1": {"strength_score": 0.75}}},
        "uncertainty_analysis": {"overall_uncertainty": 0.2}
    }
    
    test_industry_chain = {
        "industry_name": "新能源汽车",
        "companies": [
            {"name": "特锐德", "code": "300001", "business": "充电桩制造", "position": "下游"},
            {"name": "科士达", "code": "002518", "business": "充电设备", "position": "下游"}
        ]
    }
    
    # 生成投资建议 - Generate investment recommendations
    try:
        recommendations = advisor.generate_recommendations(test_causal_graph, test_industry_chain)
        
        if recommendations:
            print("✓ 投资建议生成成功")
            for rec in recommendations:
                print(f"  股票: {rec.get('stock_name')} ({rec.get('stock_code')})")
                print(f"  建议: {advisor.advice_types.get(rec.get('advice_type'))}")
                print(f"  风险: {rec.get('risk_level_cn')}")
            
            # 生成报告 - Generate report
            report = advisor.generate_report({
                "investment_recommendations": recommendations,
                # ... 其他分析结果 ...
            })
            print("\n文本报告预览:")
            print(report[:300] + "...")
            
        else:
            print("✗ 未生成投资建议")
            
    except Exception as e:
        print(f"✗ 测试异常: {e}")
    
    print("测试完成")
