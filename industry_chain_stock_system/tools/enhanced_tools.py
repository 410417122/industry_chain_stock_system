"""
增强型工具模块
Enhanced Tools Module

包含搜索工具和基于AKShare的数据获取工具
Contains search tools and data acquisition tools based on AKShare
"""

import json
from typing import List, Dict, Any, Optional
from crewai.tools import BaseTool
from pydantic import ConfigDict, Field
import akshare as ak
import pandas as pd
from pathlib import Path
import sys
import requests # requests 可能在某些 akshare 函数间接依赖或用户自定义工具中使用
import base64
from datetime import datetime, timedelta

# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# SerperSearchTool 将被 crewai_tools.SerperDevTool 替代，此类将被移除或重构。
# 为了最小化当前更改的范围，我们先保留 IndustrySearchTool 和 RelevanceAnalysisTool,
# 并修改它们的构造函数以接受一个标准的 BaseTool 实例（如 SerperDevTool）。

class IndustrySearchTool(BaseTool):
    """
    行业搜索工具
    Industry Search Tool
    
    用于搜索特定行业的信息，包括政策、趋势、龙头企业等。
    它内部使用一个通用的搜索工具来执行实际的搜索。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "行业研究工具"
    description: str = ("用于搜索特定行业的信息，包括产业链结构、政策、趋势、龙头企业等。"
                       "输入行业名称和搜索类型（如 'overview', 'policy', 'trend', 'leaders', 'chain'），返回相关信息的摘要。")
    search_tool: BaseTool = Field(..., description="一个实现了run方法的搜索工具实例 (例如 crewai_tools.SerperDevTool)。")

    def _run(self, industry: str, search_type: str = "overview") -> str:
        """
        执行行业搜索
        Execute industry search
        
        Args:
            industry: 行业名称
            search_type: 搜索类型，可选 "overview"、"policy"、"trend"、"leaders"、"chain"
            
        Returns:
            str: 搜索结果的JSON字符串
        """
        try:
            # 根据搜索类型构建查询
            if search_type == "overview":
                query = f"{industry} 行业概况 产业结构"
            elif search_type == "policy":
                query = f"{industry} 行业政策 最新政策 监管"
            elif search_type == "trend":
                query = f"{industry} 行业趋势 发展前景 最新动态"
            elif search_type == "leaders":
                query = f"{industry} 行业龙头企业 市场份额 代表企业"
            elif search_type == "chain":
                query = f"{industry} 产业链 上下游 供应链结构"
            else:
                query = f"{industry} {search_type}"
            
            search_output_dict = self.search_tool.run(search_query=query)

            if not isinstance(search_output_dict, dict):
                try:
                    parsed_if_error_str = json.loads(str(search_output_dict))
                    if isinstance(parsed_if_error_str, dict) and "error" in parsed_if_error_str:
                        return json.dumps(parsed_if_error_str, ensure_ascii=False)
                except:
                    pass
                return json.dumps({"error": f"IndustrySearchTool: Expected a dictionary from search_tool.run(), but got {type(search_output_dict)}"}, ensure_ascii=False)
            
            processed_results = []
            possible_results_keys = ['organic', 'news_results', 'results', 'items'] 
            raw_results_list = None
            for key in possible_results_keys:
                if key in search_output_dict and isinstance(search_output_dict[key], list):
                    raw_results_list = search_output_dict[key]
                    break
            
            if raw_results_list is not None:
                for item in raw_results_list:
                    if isinstance(item, dict):
                        processed_results.append({
                            "title": item.get("title", "N/A"),
                            "snippet": item.get("snippet", ""),
                            "link": item.get("link", "")
                        })
                    elif isinstance(item, str):
                        processed_results.append({"title": "Result", "snippet": item, "link": ""})
            elif isinstance(search_output_dict, dict) and not any(key in search_output_dict for key in possible_results_keys):
                 processed_results.append({
                    "title": search_output_dict.get("title", f"Search result for: {query}"),
                    "snippet": str(search_output_dict),
                    "link": search_output_dict.get("link", "")
                })

            results_package = {
                "query": query,
                "results": processed_results if processed_results else [{"title": "No parseable results", "snippet": str(search_output_dict), "link": ""}],
                "industry": industry,
                "search_type": search_type
            }
            return json.dumps(results_package, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                "error": f"行业研究工具在处理搜索结果时发生内部错误: {str(e)}",
                "industry": industry,
                "search_type": search_type
            }, ensure_ascii=False)

class StockDataTool(BaseTool):
    """
    股票数据工具
    Stock Data Tool
    
    用于获取股票数据，包括实时行情、历史数据等
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "股票数据工具"
    description: str = ("用于获取股票数据，包括实时行情、历史数据等。"
                       "输入股票代码和数据类型，返回相关数据的JSON字符串。")
    
    def _run(self, symbol: str = "", data_type: str = "real_time", 
             start_date: str = "", end_date: str = "") -> str:
        try:
            if not symbol:
                return json.dumps({"error": "股票代码不能为空"}, ensure_ascii=False)
            
            if data_type == "real_time":
                df = ak.stock_zh_a_spot_em(symbol=symbol)
                if df is not None and not df.empty:
                    stock_data = df.iloc[0].to_dict()
                    result = {"symbol": symbol, "data_type": "real_time", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "data": stock_data}
                else:
                    result = {"symbol": symbol, "data_type": "real_time", "error": "未找到股票数据"}
            elif data_type == "history":
                if not start_date: start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
                if not end_date: end_date = datetime.now().strftime("%Y%m%d")
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="")
                if df is not None and not df.empty:
                    df['日期'] = df['日期'].astype(str)
                    records = df.to_dict('records')
                    result = {"symbol": symbol, "data_type": "history", "start_date": start_date, "end_date": end_date, "data": records}
                else:
                    result = {"symbol": symbol, "data_type": "history", "start_date": start_date, "end_date": end_date, "error": "未找到历史数据"}
            elif data_type == "fundamental":
                try:
                    df = ak.stock_financial_analysis_indicator(symbol=symbol)
                    if df is not None and not df.empty:
                        latest_data = df.iloc[0].to_dict()
                        result = {"symbol": symbol, "data_type": "fundamental", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "data": latest_data}
                    else:
                        result = {"symbol": symbol, "data_type": "fundamental", "error": "未找到基本面数据"}
                except:
                    try:
                        df = ak.stock_individual_info_em(symbol=symbol)
                        if df is not None and not df.empty:
                            info_data = {row.iloc[0]: row.iloc[1] for _, row in df.iterrows()}
                            result = {"symbol": symbol, "data_type": "fundamental", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "data": info_data}
                        else:
                            result = {"symbol": symbol, "data_type": "fundamental", "error": "未找到基本面数据"}
                    except Exception as e_fund:
                        result = {"symbol": symbol, "data_type": "fundamental", "error": f"获取基本面数据失败: {str(e_fund)}"}
            else:
                result = {"symbol": symbol, "error": f"不支持的数据类型: {data_type}"}
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"symbol": symbol, "data_type": data_type, "error": f"获取股票数据失败: {str(e)}"}, ensure_ascii=False)

class IndustryStockTool(BaseTool):
    """
    行业股票工具
    Industry Stock Tool
    
    用于获取特定行业的股票列表和行业分类信息
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "行业股票工具"
    description: str = ("用于获取特定行业的股票列表和行业分类信息。"
                       "输入行业代码或名称，返回相关股票的列表和信息。")
    
    def _run(self, industry: str, classification: str = "sw", detail_level: str = "1") -> str:
        try:
            result = {}
            try:
                df = ak.stock_board_industry_cons_em(symbol=industry)
            except Exception as e:
                return json.dumps({
                    "industry": industry,
                    "classification": classification,
                    "detail_level": detail_level,
                    "error": f"使用ak.stock_board_industry_cons_em获取'{industry}'的成分股失败: {str(e)}. 请确保行业名称是AKShare可识别的板块名称。",
                    "suggestion": "您可以尝试使用【行业研究工具】搜索 'AKShare板块列表' 或 '申万行业分类' 以查找有效的板块名称，或尝试更通用/标准的行业名称（例如，对于申万行业，尝试如 '计算机' 而非 '计算机应用'）。"
                }, ensure_ascii=False)

            if df is not None and not df.empty:
                stocks = df.to_dict('records')
                result = {"industry": industry, "classification": classification, "detail_level": detail_level, "stock_count": len(stocks), "stocks": stocks}
            else:
                result = {"industry": industry, "classification": classification, "detail_level": detail_level, "error": f"AKShare未能为板块'{industry}'返回任何股票数据。可能原因：1. 该板块名称不被AKShare识别。2. 该板块下当前没有成分股。请尝试使用更通用或标准的行业名称，或使用行业研究工具确认板块名称。"}
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"industry": industry, "classification": classification, "detail_level": detail_level, "error": f"获取行业股票数据失败: {str(e)}"}, ensure_ascii=False)

class StockScreeningTool(BaseTool):
    """股票筛选工具"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "股票筛选工具"
    description: str = ("用于根据特定条件筛选股票。输入筛选条件字典，返回符合条件的股票列表。")
    
    def _run(self, conditions: Dict[str, Any] = None) -> str:
        try:
            if conditions is None: conditions = {}
            df = ak.stock_zh_a_spot_em()
            filtered_df = df.copy()
            if "industry" in conditions and conditions["industry"]:
                try:
                    industry_df = ak.stock_sector_spot(sector=conditions["industry"])
                    if industry_df is not None and not industry_df.empty:
                        industry_codes = industry_df["代码"].tolist()
                        filtered_df = filtered_df[filtered_df["代码"].isin(industry_codes)]
                except: pass
            if "market_cap_min" in conditions and conditions["market_cap_min"] is not None: filtered_df = filtered_df[filtered_df["总市值"] >= conditions["market_cap_min"] * 100000000]
            if "market_cap_max" in conditions and conditions["market_cap_max"] is not None: filtered_df = filtered_df[filtered_df["总市值"] <= conditions["market_cap_max"] * 100000000]
            if "pe_min" in conditions and conditions["pe_min"] is not None: filtered_df = filtered_df[filtered_df["市盈率-动态"] >= conditions["pe_min"]]
            if "pe_max" in conditions and conditions["pe_max"] is not None: filtered_df = filtered_df[filtered_df["市盈率-动态"] <= conditions["pe_max"]]
            if "pb_min" in conditions and conditions["pb_min"] is not None: filtered_df = filtered_df[filtered_df["市净率"] >= conditions["pb_min"]]
            if "pb_max" in conditions and conditions["pb_max"] is not None: filtered_df = filtered_df[filtered_df["市净率"] <= conditions["pb_max"]]
            limit = conditions.get("limit", 50)
            if limit > 0: filtered_df = filtered_df.head(limit)
            if not filtered_df.empty:
                stocks = filtered_df.to_dict('records')
                result = {"conditions": conditions, "stock_count": len(stocks), "stocks": stocks}
            else:
                result = {"conditions": conditions, "stock_count": 0, "stocks": [], "message": "未找到符合条件的股票"}
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"conditions": conditions, "error": f"股票筛选失败: {str(e)}"}, ensure_ascii=False)

class RelevanceAnalysisTool(BaseTool):
    """相关性分析工具"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "相关性分析工具"
    description: str = ("用于分析股票与事件/行业的相关性，将股票分为强相关、中相关和弱相关三类。"
                       "输入事件描述、行业和股票列表，返回相关性分析结果。")
    search_tool: BaseTool = Field(..., description="一个实现了run方法的搜索工具实例 (例如 crewai_tools.SerperDevTool)。")

    def _run(self, event_description: str, industry: str, stocks: List[Dict[str, Any]]) -> str:
        try:
            if not event_description: return json.dumps({"error": "事件描述不能为空"}, ensure_ascii=False)
            if not industry: return json.dumps({"error": "行业名称不能为空"}, ensure_ascii=False)
            if not stocks: return json.dumps({"error": "股票列表不能为空"}, ensure_ascii=False)
            strong_related, medium_related, weak_related = [], [], []
            
            chain_search_output = self.search_tool.run(search_query=f"{industry} 产业链 上游 中游 下游")
            if isinstance(chain_search_output, str): chain_info = json.loads(chain_search_output)
            elif isinstance(chain_search_output, dict): chain_info = chain_search_output
            else: chain_info = {"error": f"Unexpected search result type for chain_info: {type(chain_search_output)}"}

            for stock in stocks:
                stock_code, stock_name = stock.get("代码", ""), stock.get("名称", "")
                if not stock_code or not stock_name: continue
                relevance_score = 50
                
                event_query = f"{stock_name} {event_description}"
                event_search_output = self.search_tool.run(search_query=event_query)
                if isinstance(event_search_output, str): event_results = json.loads(event_search_output)
                elif isinstance(event_search_output, dict): event_results = event_search_output
                else: event_results = {"error": "Unexpected search result type", "results": []}
                
                event_relevance = 0
                for item in event_results.get("results", []):
                    title, snippet = item.get("title", "").lower(), item.get("snippet", "").lower()
                    if stock_name.lower() in title and any(kw in title for kw in event_description.split()): event_relevance += 10
                    elif stock_name.lower() in snippet and any(kw in snippet for kw in event_description.split()): event_relevance += 5
                relevance_score += min(event_relevance, 20)
                
                position_query = f"{stock_name} {industry} 产业链 位置"
                position_search_output = self.search_tool.run(search_query=position_query)
                if isinstance(position_search_output, str): position_results = json.loads(position_search_output)
                elif isinstance(position_search_output, dict): position_results = position_search_output
                else: position_results = {"error": "Unexpected search result type", "results": []}
                
                position_text = "".join(f"{item.get('title', '')} {item.get('snippet', '')}" for item in position_results.get("results", []))
                position_relevance = 0
                if any(k in position_text for k in ["龙头", "领军", "头部"]): position_relevance += 15
                elif any(k in position_text for k in ["核心", "重要"]): position_relevance += 10
                if "上游" in position_text: position_relevance += 5
                elif "中游" in position_text: position_relevance += 8
                elif "下游" in position_text: position_relevance += 5
                relevance_score += min(position_relevance, 20)
                
                business_query = f"{stock_name} 主营业务"
                business_search_output = self.search_tool.run(search_query=business_query)
                if isinstance(business_search_output, str): business_results = json.loads(business_search_output)
                elif isinstance(business_search_output, dict): business_results = business_search_output
                else: business_results = {"error": "Unexpected search result type", "results": []}

                business_text = "".join(f"{item.get('title', '')} {item.get('snippet', '')}" for item in business_results.get("results", []))
                keyword_match_count = sum(1 for keyword in event_description.split() if keyword in business_text)
                relevance_score += min(10 * keyword_match_count, 15) if keyword_match_count > 0 else 0
                
                stock_result = {"code": stock_code, "name": stock_name, "relevance_score": relevance_score, "event_relevance": event_relevance, "position_relevance": position_relevance, "business_relevance": min(10 * keyword_match_count, 15)}
                if relevance_score >= 80: strong_related.append(stock_result)
                elif relevance_score >= 60: medium_related.append(stock_result)
                else: weak_related.append(stock_result)
            
            sort_key = lambda x: x["relevance_score"]
            result_summary = {
                "event": event_description, "industry": industry, "total_stocks": len(stocks),
                "strong_related": {"count": len(strong_related), "stocks": sorted(strong_related, key=sort_key, reverse=True)},
                "medium_related": {"count": len(medium_related), "stocks": sorted(medium_related, key=sort_key, reverse=True)},
                "weak_related": {"count": len(weak_related), "stocks": sorted(weak_related, key=sort_key, reverse=True)}
            }
            return json.dumps(result_summary, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"event": event_description, "industry": industry, "error": f"相关性分析失败: {str(e)}"}, ensure_ascii=False)

class MacroDataTool(BaseTool):
    """宏观数据工具"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "宏观数据工具"
    description: str = ("用于获取宏观经济数据，包括PMI、GDP、CPI等。"
                       "输入数据类型，返回相关数据的JSON字符串。")
    
    def _run(self, data_type: str = "pmi") -> str:
        try:
            processed_data_type = data_type.lower()
            df = None
            data_name, unit = "", ""

            if processed_data_type == "pmi":
                df = ak.macro_china_pmi()
                data_name, unit = "中国制造业采购经理指数(PMI)", "%"
            elif processed_data_type == "gdp":
                df = ak.macro_china_gdp_yearly()
                data_name, unit = "中国GDP年度增速", "%"
            elif processed_data_type == "cpi":
                df = ak.macro_china_cpi_yearly()
                data_name, unit = "中国CPI年度数据", "%"
            elif processed_data_type == "interest_rate":
                df = ak.macro_china_lpr()
                data_name, unit = "中国贷款市场报价利率(LPR)", "%"
            elif processed_data_type == "exchange_rate":
                df = ak.macro_china_fx_reserves()
                data_name, unit = "中国外汇储备", "亿美元"
            else:
                return json.dumps({"error": f"不支持的数据类型: {data_type}"}, ensure_ascii=False)

            if df is not None and not df.empty:
                if df.columns[0] == '日期' or '日期' in df.columns or df.iloc[:, 0].dtype == 'datetime64[ns]' or pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
                     df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0]).astype(str) # More robust date conversion
                records = df.to_dict('records')
                result = {"data_type": processed_data_type, "name": data_name, "unit": unit, "data": records}
            else:
                result = {"data_type": processed_data_type, "error": f"未找到{data_name}数据"}
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"data_type": data_type, "error": f"获取宏观数据失败: {str(e)}"}, ensure_ascii=False)

class ListIndustrySectorsTool(BaseTool):
    """
    可识别行业板块列表获取工具
    Tool to fetch a list of recognizable industry/sector names.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "可识别行业板块列表获取工具"
    description: str = (
        "根据指定的数据源获取AKShare可识别的行业或概念板块名称列表。"
        "有效的 'source' 包括: "
        "'sw' (申万行业分类), "
        "'ths' (同花顺行业分类), "
        "'em_industry' (东方财富行业板块), "
        "'em_concept' (东方财富概念板块)."
    )

    def _run(self, source: str = "sw") -> str:
        """
        获取行业或概念板块名称列表。
        Fetches a list of industry or concept sector names.

        Args:
            source (str): 数据源。默认为 "sw"。
                          可选: "sw", "ths", "em_industry", "em_concept".
                          The data source. Defaults to "sw".
                          Options: "sw", "ths", "em_industry", "em_concept".
        
        Returns:
            str: 包含板块名称列表的JSON字符串，或错误信息。
                 A JSON string containing a list of sector names, or an error message.
        """
        source = source.lower()
        sectors = []
        error_message = None
        try:
            if source == "sw":
                df_sw = ak.get_industry_category(source="sw")
                for col in df_sw.columns:
                    if 'name' in col.lower() or 'industry' in col.lower(): # 修正冒号
                        sectors.extend(df_sw[col].dropna().unique().tolist())
                sectors = sorted(list(set(s for s in sectors if isinstance(s, str) and s.strip())))
                if not sectors: 
                    df_sw_l1 = ak.sw_index_first_info()
                    sectors = df_sw_l1['行业名称'].dropna().unique().tolist()

            elif source == "ths":
                df_ths = ak.get_industry_category(source="ths")
                for col in df_ths.columns:
                     if 'name' in col.lower() or 'industry' in col.lower(): # 修正冒号
                        sectors.extend(df_ths[col].dropna().unique().tolist())
                sectors = sorted(list(set(s for s in sectors if isinstance(s, str) and s.strip())))
            
            elif source == "em_industry":
                df_em_industry = ak.stock_board_industry_name_em()
                sectors = df_em_industry["板块名称"].dropna().unique().tolist()
            
            elif source == "em_concept":
                df_em_concept = ak.stock_board_concept_name_em()
                sectors = df_em_concept["板块名称"].dropna().unique().tolist()
            
            else:
                error_message = f"不支持的数据源: '{source}'. 有效来源为 'sw', 'ths', 'em_industry', 'em_concept'."
                
            if error_message:
                return json.dumps({"error": error_message, "requested_source": source}, ensure_ascii=False)
            
            if not sectors and not error_message: 
                 error_message = f"未能从数据源 '{source}' 获取到任何板块名称。"
                 return json.dumps({"error": error_message, "requested_source": source, "sectors": []}, ensure_ascii=False)

            return json.dumps({"source": source, "count": len(sectors), "sectors": sectors}, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": f"获取板块列表时发生错误 (源: {source}): {str(e)}", "requested_source": source}, ensure_ascii=False)
