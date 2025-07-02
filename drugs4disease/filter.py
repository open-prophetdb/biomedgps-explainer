import pandas as pd
import re

class DrugFilter:
    """
    负责解析用户表达式并对DataFrame进行筛选。
    """
    @staticmethod
    def parse_expression(expr: str) -> str:
        """
        将用户表达式中的 '&&' 和 '||' 替换为 pandas.query 支持的 'and' 和 'or'。
        """
        expr = expr.replace('&&', 'and').replace('||', 'or')
        return expr

    @staticmethod
    def filter_dataframe(df: pd.DataFrame, expr: str) -> pd.DataFrame:
        """
        根据表达式筛选DataFrame。
        """
        query_expr = DrugFilter.parse_expression(expr)
        return df.query(query_expr) 