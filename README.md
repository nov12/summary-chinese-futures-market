# 期货市场近期高低点汇总表

## 功能说明

本项目用于生成中国期货市场的近期高低点汇总数据，支持以下输出格式：

1. **Markdown格式** - 通过邮件发送
2. **Vue.js美化的可排序HTML网页** - 本地保存为HTML文件

## 新增功能：Vue.js可排序表格

### 特性

- ✅ 使用Vue.js框架生成美观的HTML表格
- ✅ 点击任意列标题即可对该列进行排序
- ✅ 支持升序/降序切换
- ✅ 自动识别数字和文本，智能排序
- ✅ 响应式设计，适配不同屏幕尺寸
- ✅ 清晰的排序指示器（▲ 升序，▼ 降序）

### 使用方法

运行主程序后，会自动生成两种输出：

1. **邮件发送**：继续保持原有的Markdown格式邮件发送功能
2. **HTML文件**：在项目根目录生成 `futures_summary_table.html` 文件

```python
# 在main.py中，程序会自动：
# 1. 生成Markdown并发送邮件
# 2. 生成Vue.js HTML文件
vue_html = email.convert2vue_html(df, '期货市场近期高低点汇总表')
with open('futures_summary_table.html', 'w', encoding='utf-8') as f:
    f.write(vue_html)
```

### 查看生成的HTML

在浏览器中打开 `futures_summary_table.html` 文件即可查看美化后的可排序表格。

### 示例截图

**初始状态**
![初始表格](https://github.com/user-attachments/assets/40c6b2b0-247d-4484-b385-f5a58434a895)

**升序排序**
![升序排序](https://github.com/user-attachments/assets/8480360f-5931-4691-ae90-87f92cce530e)

**降序排序**
![降序排序](https://github.com/user-attachments/assets/251075c8-1970-4ba7-b6b2-9bdf9e2f9b8c)

## API说明

### HtmlEmail.convert2vue_html()

将DataFrame转换为使用Vue.js的可排序HTML表格。

**参数：**
- `df` (pd.DataFrame): 要转换的数据框
- `title` (str): HTML页面标题，默认为 '期货市场近期高低点汇总表'

**返回：**
- `str`: 完整的HTML字符串，包含Vue.js代码和样式

**示例：**
```python
from htmlemail import HtmlEmail
import pandas as pd

email = HtmlEmail(account, password, smtp_host, smtp_port)
html_content = email.convert2vue_html(df, '期货数据表')
```

## 依赖项

无需新增Python依赖。生成的HTML使用Vue.js的CDN版本（jsdelivr）。

**注意事项：**
- 生成的HTML文件需要网络连接才能加载Vue.js
- 如果需要离线使用，可以下载Vue.js到本地并修改script标签路径
- 本功能生成的是本地HTML文件，仅供查看使用

## 兼容性

- ✅ 保持向后兼容
- ✅ 原有的Markdown邮件功能不受影响
- ✅ 支持所有现代浏览器
