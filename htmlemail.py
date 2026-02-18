import smtplib
from email.header import Header
from email.mime.text import MIMEText
from typing import Union

import markdown
import pandas as pd
from tabulate import tabulate


class HtmlEmail:
    """
    将markdown转换为html邮件发送
    """

    def __init__(self, account: str, password: str, smtp_server: str, smtp_port: int):
        self.sender = account
        self._password = password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def send_html(self, receivers: Union[list, str], subject: str, html_content: str):
        """
        Sends an email with the specified subject and HTML content.
        """
        if not isinstance(receivers, list):
            receivers = [receivers]

        msg = MIMEText(html_content, 'html', 'utf-8')
        msg['From'] = Header(self.sender)
        msg['Subject'] = Header(subject)

        server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
        server.login(self.sender, self._password)
        for receiver in receivers:
            msg['To'] = Header(receiver)
            server.sendmail(self.sender, receiver, msg.as_string())
        server.quit()

    def convert2html(self, markdown_str: str) -> str:
        """
        将Markdown转换为HTML
        """
        # 将Markdown格式的字符串转换为HTML
        extensions = ['markdown.extensions.tables']  # 添加表格扩展
        html = markdown.markdown(markdown_str, extensions=extensions)

        # 使用CSS样式为表格添加内框线和外框线
        css_style = '''
            <style>
            table {
                border-collapse: collapse;
                border: 1px solid black;
                border-spacing: 0;
            }
            th, td {
                border: 1px solid black;
                padding: 5px;
            }
            </style>
        '''
        html = css_style + html

        return html

    def convert2md(self, df: pd.DataFrame) -> str:
        """
        将DataFrame转换为Markdown
        """
        # 将headers加粗
        headers = ['**' + header + '**' for header in df.columns]

        markdown_table = tabulate(df, headers=headers, tablefmt='pipe', showindex=False)
        return markdown_table

    def convert2vue_html(self, df: pd.DataFrame, title: str = '期货市场近期高低点汇总表') -> str:
        """
        将DataFrame转换为使用Vue.js的可排序HTML表格
        """
        # 将DataFrame转换为JSON格式的数据
        data_json = df.to_json(orient='records', force_ascii=False)
        
        # 获取列名
        columns = list(df.columns)
        
        # 生成HTML
        html_template = f'''<!DOCTYPE html>
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
</html>'''
        
        return html_template
