#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试Vue.js HTML生成功能
"""
import pandas as pd
from htmlemail import HtmlEmail


def test_convert2vue_html():
    """测试convert2vue_html方法"""
    # 创建一个测试DataFrame
    test_data = {
        '合约': ['rb2405', 'hc2405', 'i2405'],
        '方向': ['多', '空', '多'],
        '天数': [90, 120, 150],
        '距离%': [2.5, 3.8, 1.2],
        '今日最高': [3850, 3620, 890],
        '365天内最高': [4200, 3900, 950],
        '今日最低': [3800, 3580, 880],
        '365天内最低': [3500, 3300, 800]
    }
    df = pd.DataFrame(test_data)
    
    # 创建HtmlEmail实例（不需要实际的邮件配置来测试）
    email = HtmlEmail('test@test.com', 'test_password', 'smtp.test.com', 465)
    
    # 生成Vue.js HTML
    vue_html = email.convert2vue_html(df, '测试期货市场汇总表')
    
    # 验证生成的HTML包含必要的元素
    assert '<!DOCTYPE html>' in vue_html
    assert 'Vue' in vue_html
    assert '测试期货市场汇总表' in vue_html
    assert 'rb2405' in vue_html
    assert 'hc2405' in vue_html
    assert 'i2405' in vue_html
    assert 'sortBy' in vue_html
    assert 'sortedData' in vue_html
    
    # 保存HTML文件进行手动检查
    with open('test_output.html', 'w', encoding='utf-8') as f:
        f.write(vue_html)
    
    print('✓ Vue.js HTML生成测试通过')
    print('✓ 测试文件已保存为 test_output.html')
    print('✓ 请在浏览器中打开测试文件验证排序功能')
    
    return True


if __name__ == '__main__':
    test_convert2vue_html()
