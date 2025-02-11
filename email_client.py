import smtplib
from email.header import Header
from email.mime.text import MIMEText
from typing import Union


class Email:
    """
    将markdown转换为html邮件发送
    """

    def __init__(self, account: str, password: str, smtp_server: str, smtp_port: int) -> None:
        self.sender = account
        self._password = password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def send_html(self, receivers: Union[list, str], subject: str, html_content: str) -> None:
        """
        Sends an email with the specified subject and HTML content.
        """
        if not isinstance(receivers, list):
            receivers = [receivers]

        msg = MIMEText(html_content, "html", "utf-8")
        msg["From"] = Header(self.sender)
        msg["Subject"] = Header(subject)

        server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
        server.login(self.sender, self._password)
        for receiver in receivers:
            msg["To"] = Header(receiver)
            server.sendmail(self.sender, receiver, msg.as_string())
        server.quit()
