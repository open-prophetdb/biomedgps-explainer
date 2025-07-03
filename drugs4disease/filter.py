import pandas as pd
import re

class DrugFilter:
    """
    Responsible for parsing user expressions and filtering the DataFrame.
    """
    @staticmethod
    def parse_expression(expr: str) -> str:
        """
        Replace '&&' and '||' in user expressions with 'and' and 'or' supported by pandas.query.
        """
        expr = expr.replace('&&', 'and').replace('||', 'or')
        return expr

    @staticmethod
    def filter_dataframe(df: pd.DataFrame, expr: str) -> pd.DataFrame:
        """
        Filter the DataFrame based on the expression.
        """
        query_expr = DrugFilter.parse_expression(expr)
        return df.query(query_expr) 

    def filter_drugs(
        self,
        input_file: str,
        expression: str,
        output_file: str,
        sheet_names: tuple[str, str] = ("annotated_drugs", "filtered_drugs")
    ) -> None:
        """
        Filter drugs based on the expression.
        """
        df = pd.read_excel(input_file, sheet_name=sheet_names[0])
        filtered_df = self.filter_dataframe(df, expression)
        
        with pd.ExcelWriter(output_file) as writer:
            df.to_excel(writer, sheet_name=sheet_names[0], index=False)
            filtered_df.to_excel(writer, sheet_name=sheet_names[1], index=False)
