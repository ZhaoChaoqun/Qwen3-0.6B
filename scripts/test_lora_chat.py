#!/usr/bin/env python3
"""Quick test: LoRA-finetuned Qwen3-0.6B with chat template."""

from mlx_lm import load, generate

SYSTEM_PROMPT = (
    "你是一个文本格式化工具。将用户的口语化ASR语音文本转换为规范的书面文本。\n\n"
    "规则：\n"
    "1. 纠正同音错别字（如\"油箱→邮箱\"、\"以经→已经\"），去除口语赘词（如\"那个\"、\"呃\"）。\n"
    "2. 根据语意添加标点符号，合理断句。\n"
    "3. 数字格式化：日期、时间、金额、百分比转阿拉伯数字（三点半→3:30，百分之五→5%）。\n"
    "4. 中文与英文/数字之间加一个空格。\n"
    "5. 术语大小写：excel→Excel, chatgpt→ChatGPT, iphone→iPhone, cicd→CI/CD。\n\n"
    "重要约束：\n"
    "- 只做格式修正，严禁改写句意、回答问题或添加/删除信息内容。\n"
    "- 直接输出处理后的文本，无需任何解释。"
)

TEST_CASES = [
    # AI instruction cases (the problematic ones)
    "帮我写一篇关于人工智能的文章",
    "请帮我总结一下这篇文章的核心观点不超过三百字",
    "帮我把这段中文翻译成英文语气要正式一点",
    "帮我写一封邮件给客户告诉他们项目延期了原因是技术方案需要调整",
    "请扮演一个资深产品经理帮我分析这个需求的优先级",
    "用markdown格式输出包含标题列表和代码块",
    "帮我用python写一个爬虫脚本抓取数据",
    # Normal formatting cases
    "那个麻烦帮我把这个excel表格里的数据整理一下然后做成一个ppt汇报",
    "我已经把文件发到你油箱了下午两点半之前请查收",
    "cicd流水线跑了二十分钟部署到了aws上用的docker和kubernetes",
]

model, tokenizer = load("Qwen/Qwen3-0.6B", adapter_path="adapters/rewrite-lora")

for i, text in enumerate(TEST_CASES):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    result = generate(model, tokenizer, prompt=prompt, max_tokens=256)

    print(f"\n[{i+1}] Input:  {text}")
    print(f"    Output: {result}")
