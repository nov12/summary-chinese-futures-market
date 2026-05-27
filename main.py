from pathlib import Path

import yaml

from .converter import Coverter
from .email_client import Email
from .tqdata import TqdataClient

if __name__ == "__main__":

    path = Path("/etc/scfm_config.yaml")
    if not path.exists():
        path = Path("./config.yaml")

    # 读取配置文件
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tq = TqdataClient(
        config["tqsdk"]["username"],
        config["tqsdk"]["password"],
        intervals=config["tqsdk"]["intervals"],
    )

    tq.login()
    tq.query_contracts()
    tq.query_all_history()
    df = tq.generate_extreme_dataframe()
    tq.api.close()

    # 生成Markdown格式的字符串
    markdown_str = Coverter.df2md(df)

    # 将Markdown格式的字符串转换为HTML
    html = Coverter.md2html(markdown_str)

    # 为HTML增加网页头部的信息和表格样式以及标题
    html = Coverter.add_html_header(html, "期货市场近期高低点汇总表")

    # 为HTML增加时间戳
    html = Coverter.add_timestamp(html)

    # 生成Vue.js美化的可排序HTML页面
    vue_html = Coverter.vue_html(df, '期货市场近期高低点汇总表')
    
    # # 保存为HTML文件
    # with open('futures_summary_table.html', 'w', encoding='utf-8') as f:
    #     f.write(vue_html)
    # print('Vue.js HTML文件已生成: futures_summary_table.html')

    # 发送邮件
    email.send_html(receivers, '期货市场近期高低点汇总表', vue_html)
    tq.api.close()
