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
