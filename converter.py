import datetime

import markdown
import pandas as pd
from tabulate import tabulate


class Coverter:
    """
    将Markdown转换为HTML或将DataFrame转换为Markdown
    """

    @staticmethod
    def md2html(markdown_str: str) -> str:
        """
        将Markdown转换为HTML
        """
        # 将Markdown格式的字符串转换为HTML
        extensions = ["markdown.extensions.tables"]  # 添加表格扩展
        html = markdown.markdown(markdown_str, extensions=extensions)

        return html

    @staticmethod
    def add_html_header(html: str, title: str) -> str:
        """
        为HTML增加网页头部的信息和表格样式以及标题
        """
        head = """
        <head>
            <title>{title}</title>
            <style>
                table {{
                    border-collapse: collapse;
                    border: 1px solid black;
                    border-spacing: 0;
                }}
                th, td {{
                    border: 1px solid black;
                    padding: 5px;
                }}
        </head>
        """.format(
            title=title
        )
        body = f"<body><h1>{title}</h1>{html}</body>"
        html = f"<html>{head}{body}</html>"

        return html

    @staticmethod
    def add_timestamp(html: str) -> str:
        """
        为HTML增加时间戳
        """
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_stamp = f"<div style='text-align: right;'>制表时间: {now}</div>"
        if "</body>" in html:
            html = html.replace("</body>", f"{time_stamp}</body>")
        else:
            html = f"{html}{time_stamp}"

    @staticmethod
    def df2md(df: pd.DataFrame) -> str:
        """
        将DataFrame转换为Markdown
        """
        # 将headers加粗
        headers = ["**" + header + "**" for header in df.columns]

        markdown_table = tabulate(df, headers=headers, tablefmt="pipe", showindex=False)
        return markdown_table
