import datetime
import re

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
        如果存在head内容则替换，包括标题
        """
        head = """
        <head>
            <title>{title}</title>
            <style>
                .container {{
                    margin: 0 auto;
                    width: fit-content;
                }}
                table {{
                    border-collapse: collapse;
                    border: 1px solid black;
                    border-spacing: 0;
                    margin: 0;
                }}
                th, td {{
                    border: 1px solid black;
                    padding: 5px;
                }}
                h1 {{
                    text-align: center;
                }}
                .timestamp {{
                    text-align: left;
                }}
            </style>
        </head>
        """.format(
            title=title
        )
        body = f'<body><div class="container"><h1>{title}</h1>{html}</div></body>'

        # 检查是否已经存在 <head> 标签
        if "<head" in html:
            # 使用正则表达式替换 <head> 标签及其内容
            new_html = re.sub(r"<head.*?</head>", head, html, flags=re.DOTALL)
            # 替换 <h1> 标签中的标题
            new_html = re.sub(r"<h1>.*?</h1>", f"<h1>{title}</h1>", new_html)
        else:
            new_html = f"<html>{head}{body}</html>"

        return new_html

    @staticmethod
    def add_timestamp(html: str) -> str:
        """
        为HTML增加时间戳
        """
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_stamp = f'<div class="timestamp">制表时间: {now}</div>'
        if "</h1>" in html:
            html = html.replace("</h1>", f"</h1>{time_stamp}")
        else:
            html = f"{html}{time_stamp}"
        return html

    @staticmethod
    def df2md(df: pd.DataFrame) -> str:
        """
        将DataFrame转换为Markdown
        """
        # 将headers加粗
        headers = ["**" + header + "**" for header in df.columns]

        markdown_table = tabulate(df, headers=headers, tablefmt="pipe", showindex=False)
        return markdown_table
