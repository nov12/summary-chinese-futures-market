from htmlemail import HtmlEmail
from tqdata import TqdataClient
import yaml


if __name__ == '__main__':

    # 读取配置文件
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tq = TqdataClient(config['tq']['username'], config['tq']
                      ['password'], intervals=config['tq']['intervals'])
    email = HtmlEmail(config['email']['account'], config['email']['password'],
                      config['email']['smtp_host'], config['email']['smtp_port'])
    receivers = config['email']['receivers']

    tq.login()
    tq.query_contracts()
    tq.query_all_history()
    df = tq.generate_extreme_dataframe()

    # 生成Markdown格式的字符串
    markdown_str = email.convert2md(df)
    # 将Markdown格式的字符串转换为HTML
    html = email.convert2html(markdown_str)
    # 发送邮件
    email.send_html(receivers, '期货市场近期高低点汇总表', html)
    tq.api.close()
