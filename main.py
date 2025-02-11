from converter import Coverter
from email_client import Email
from tqdata import TqdataClient
import yaml


if __name__ == "__main__":

    # 读取配置文件
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tq = TqdataClient(
        config["tq"]["username"], config["tq"]["password"], intervals=config["tq"]["intervals"]
    )
    if config["email"]["enable"]:
        email = Email(
            config["email"]["account"],
            config["email"]["password"],
            config["email"]["smtp_host"],
            config["email"]["smtp_port"],
        )
    receivers = config["email"]["receivers"]

    tq.login()
    tq.query_contracts()
    tq.query_all_history()
    df = tq.generate_extreme_dataframe()
    tq.api.close()

    # 生成Markdown格式的字符串
    markdown_str = Coverter.convert2md(df)

    # 将Markdown格式的字符串转换为HTML
    html = Coverter.convert2html(markdown_str)

    # 为HTML增加网页头部的信息和表格样式以及标题
    html = Coverter.add_html_header(html, "期货市场近期高低点汇总表")

    # 为HTML增加时间戳
    html = Coverter.add_timestamp(html)

    # 发送邮件
    if config["email"]["enable"]:
        email.send_html(receivers, "期货市场近期高低点汇总表", html)

    # 保存为HTML文件
    if config["html"]["enable"]:
        with open("index.html", "w") as f:
            f.write(html)
