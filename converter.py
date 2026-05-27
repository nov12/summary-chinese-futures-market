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
        """.format(title=title)
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

    @staticmethod
    def vue_html(df: pd.DataFrame, title: str = "期货市场近期高低点汇总表") -> str:
        """
        将DataFrame转换为使用Vue.js的可排序HTML表格

        Args:
            df (pd.DataFrame): 要转换的数据框，包含期货市场数据
            title (str, optional): HTML页面标题. 默认为 '期货市场近期高低点汇总表'

        Returns:
            str: 完整的HTML字符串，包含Vue.js代码、样式和数据

        Example:
            >>> email = HtmlEmail('account', 'password', 'smtp.example.com', 465)
            >>> html = email.convert2vue_html(df, '期货数据表')
            >>> with open('output.html', 'w', encoding='utf-8') as f:
            ...     f.write(html)
        """
        # 将DataFrame转换为JSON格式的数据
        data_json = df.to_json(orient="records", force_ascii=False)

        # 获取列名
        columns = list(df.columns)

        # 生成HTML
        html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/vue@3/dist/vue.global.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            cursor: pointer;
            user-select: none;
            position: relative;
        }}
        th:hover {{
            background-color: #45a049;
        }}
        th.sorted-asc::after {{
            content: ' ▲';
            position: absolute;
            right: 8px;
        }}
        th.sorted-desc::after {{
            content: ' ▼';
            position: absolute;
            right: 8px;
        }}
        td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f0f0f0;
        }}
        .info {{
            text-align: center;
            color: #666;
            margin-top: 20px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div id="app" class="container">
        <h1>{title}</h1>
        <table>
            <thead>
                <tr>
                    <th v-for="column in columns" 
                        :key="column" 
                        @click="sortBy(column)"
                        :class="getSortClass(column)">
                        {{{{ column }}}}
                    </th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="(row, index) in sortedData" :key="index">
                    <td v-for="column in columns" :key="column">
                        {{{{ row[column] }}}}
                    </td>
                </tr>
            </tbody>
        </table>
        <div class="info">
            <p>点击列标题可以对该列进行排序</p>
        </div>
    </div>

    <script>
        const {{ createApp }} = Vue;

        createApp({{
            data() {{
                return {{
                    columns: {columns},
                    tableData: {data_json},
                    sortColumn: null,
                    sortDirection: 'asc'
                }}
            }},
            computed: {{
                sortedData() {{
                    if (!this.sortColumn) {{
                        return this.tableData;
                    }}

                    return [...this.tableData].sort((a, b) => {{
                        let aVal = a[this.sortColumn];
                        let bVal = b[this.sortColumn];

                        // 处理null和undefined值
                        if (aVal == null && bVal == null) return 0;
                        if (aVal == null) return this.sortDirection === 'asc' ? 1 : -1;
                        if (bVal == null) return this.sortDirection === 'asc' ? -1 : 1;

                        // 尝试转换为数字进行比较
                        const aNum = parseFloat(aVal);
                        const bNum = parseFloat(bVal);

                        if (!isNaN(aNum) && !isNaN(bNum)) {{
                            aVal = aNum;
                            bVal = bNum;
                        }}

                        if (aVal < bVal) {{
                            return this.sortDirection === 'asc' ? -1 : 1;
                        }}
                        if (aVal > bVal) {{
                            return this.sortDirection === 'asc' ? 1 : -1;
                        }}
                        return 0;
                    }});
                }}
            }},
            methods: {{
                sortBy(column) {{
                    if (this.sortColumn === column) {{
                        // 如果点击的是当前排序列，切换排序方向
                        this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
                    }} else {{
                        // 如果点击的是新列，设置为升序
                        this.sortColumn = column;
                        this.sortDirection = 'asc';
                    }}
                }},
                getSortClass(column) {{
                    if (this.sortColumn === column) {{
                        return this.sortDirection === 'asc' ? 'sorted-asc' : 'sorted-desc';
                    }}
                    return '';
                }}
            }}
        }}).mount('#app');
    </script>
</body>
</html>"""

        return html_template
