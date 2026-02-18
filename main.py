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
    
    # 生成Vue.js美化的可排序HTML页面
    vue_html = email.convert2vue_html(df, '期货市场近期高低点汇总表')
    # 保存为HTML文件
    with open('futures_summary_table.html', 'w', encoding='utf-8') as f:
        f.write(vue_html)
    print('Vue.js HTML文件已生成: futures_summary_table.html')
    
    tq.api.close()
