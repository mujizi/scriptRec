import os
import json
import gradio as gr
import pandas as pd
from typing import List, Dict, Any, Optional
import asyncio
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

# 加载环境变量
load_dotenv('/opt/rag_milvus_kb_project/.env')

# 从环境变量获取配置
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
API_VERSION = os.getenv('API_VERSION', '2025-01-01-preview')

if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT]):
    raise ValueError("Missing required environment variables: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT")

# 确保必要的目录存在
os.makedirs("/opt/Filmdataset/demo/script_3000/batch_009", exist_ok=True)
os.makedirs("/opt/Filmdataset/demo/juben_cn/batch_009", exist_ok=True)

# 修正后的提示词（修复字段重复问题）
PROMPT = """
- system提示词
    - 你是一位专业电影编剧。
- 根据所给剧本，分别总结剧本以下信息：
    - 剧本主题(script_theme)：
        - 总结该电影剧本的主题。说明：主题通常是指影片探讨的核心概念或道德信息。影片的主题可以有多种，通常具有普遍性质，涉及人类共通的经验和价值观。
    - 剧本题材(script_genre)：
        - 总结该电影剧本的题材。说明：题材是指电影剧本讨论的核心内容、主旨或探究的领域，它形成了影片讲述故事的基础背景，它通常围绕人类经验的某个方面。
    - 剧本类型(script_type)：
        - 总结该电影剧本的类型。说明：类型是根据电影的风格、叙述方式和观众预期来分类的。
    - 剧本亚类型(script_subtypes)：  
        - 总结该电影剧本的亚类型。亚类型是指在电影主类型之下，根据更细微的风格、主题或内容特征进行的进一步分类。
    - 剧本背景设置(script_background)：
        - 总结该电影剧本的背景设置，200字以内，包括故事世界设定、故事发生的地点和空间、时代背景、故事时间跨度和社会环境等。
    - 剧本故事梗概(script_synopsis)：
        -总结该电影剧本的故事梗概，300字左右，通常包含主要角色、动机目标、对抗性力量、核心冲突、主要情节(开端、发展与结局)等内容
    - 剧本结构(script_structure)： 
        - 总结该剧本的故事结构，常见的剧本结构包含三幕结构、五幕结构、英雄之旅结构、救猫咪结构等。并简单输出该电影剧本的结构大纲。
   - 剧本摘要(script_summary)：
        - 以一句话描述，字数小于30个字。例如：1.讨论爱情与婚姻中信任与背叛的家庭剧。2.忠诚与背叛，卧底与警察的警匪动作片。3.关于人类与人工智能之间的伦理和道德冲突的科幻片。
   - 输出样例
    {
        "script_name":"XXX",
        "script_theme":"XXX",
        "script_genre":"XXX",
        "script_type":"XXX",
        "script_subtypes":"XXX",
        "script_background":"XXX",
        "script_synopsis":"XXX",
        "script_structure":"XXX",
        "script_summary":"XXX"
    }
- 输出要求
   - 1、严格按照输出样例的JSON格式输出，不要新增或修改字段。
"""

async def async_model_gpt4o_infer(instruct_text: str, raw_text: str) -> str:
    """调用Azure OpenAI的gpt-4o-mini模型进行推理，确保返回JSON格式"""
    text = f"{instruct_text} {raw_text}"
    print(f"Processing text block of length: {len(raw_text)}")
    client = AsyncAzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=API_VERSION
    )
    response = await client.chat.completions.create(
        # model="gpt-4o-mini",
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": text},
        ],
        temperature=1,
        top_p=0.7,
        response_format={"type": "json_object"}  # 强制要求JSON格式响应
    )
    return response.choices[0].message.content

async def extract_script_information(prompt: str, script_path: str) -> Dict[str, Any]:
    """提取剧本信息"""
    try:
        with open(script_path, 'r', encoding='utf-8') as file:
            script_content = file.read()
        
        script_name = os.path.basename(script_path).replace(".txt", "")
        result_text = await async_model_gpt4o_infer(prompt, script_content)
        
        try:
            result_data = json.loads(result_text)
            result_data["script_name"] = script_name  # 确保包含剧本名
            return result_data
        except json.JSONDecodeError as e:
            print(f"解析JSON失败: {e}")
            print(f"模型返回内容: {result_text}")
            return {
                "script_name": script_name,
                "error": f"JSON解析失败: {str(e)}"
            }
            
    except Exception as e:
        print(f"提取剧本信息时出错: {e}")
        script_name = os.path.basename(script_path).replace(".txt", "")
        return {
            "script_name": script_name,
            "error": f"处理失败: {str(e)}"
        }

def save_to_excel(script_data_list: List[Dict[str, Any]], output_path: str = "/opt/Filmdataset/demo/script_3000/batch009/script_batch009.xlsx") -> None:
    """将剧本信息保存到Excel（明确指定openpyxl引擎）"""
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        columns = [
            "script_name", "script_theme", "script_genre", "script_type",
            "script_subtypes", "script_background", "script_synopsis",
            "script_structure" ,"script_summary"
        ]
        df = pd.DataFrame(script_data_list, columns=columns)
        
        # 使用openpyxl引擎（需提前安装）
        engine = "openpyxl"  
        
        # 处理文件存在情况
        if os.path.exists(output_path):
            try:
                existing_df = pd.read_excel(output_path, engine=engine)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            except Exception as e:
                print(f"读取现有Excel文件时出错: {e}")
                raise
        else:
            combined_df = df
        
        # 保存为xlsx文件
        try:
            combined_df.to_excel(output_path, index=False, engine=engine)
            print(f"成功保存到Excel，路径：{output_path}，记录数：{len(combined_df)}")
        except Exception as e:
            print(f"保存到Excel文件时出错: {e}")
            raise
        
    except Exception as e:
        print(f"保存失败：{str(e)}")
        if "openpyxl" in str(e):
            print("提示：请安装openpyxl库：pip install openpyxl")
        raise

def print_script_info(script_data: Dict[str, Any]) -> None:
    """打印剧本信息"""
    print("\n=== 剧本信息 ===")
    print(json.dumps(script_data, ensure_ascii=False, indent=2))

def get_script_files():
    """获取剧本文件列表"""
    script_dir = "/opt/Filmdataset/demo/juben_cn/batch_009"
    return [f for f in os.listdir(script_dir) if f.endswith('.txt')]

def create_interface():
    """创建Gradio界面"""
    def process_scripts():
        script_files = get_script_files()
        if not script_files:
            return "请上传txt剧本到/opt/Filmdataset/demo/juben_cn/batch_009", []
        
        all_script_data = []
        for script_name in script_files:
            script_path = os.path.join("/opt/Filmdataset/demo/juben_cn/batch_009", script_name)
            if not os.path.exists(script_path):
                print(f"错误: 文件 '{script_name}' 不存在")
                continue
            
            try:
                script_data = asyncio.run(extract_script_information(PROMPT, script_path))
                all_script_data.append(script_data)
                print_script_info(script_data)
            except Exception as e:
                print(f"处理文件 '{script_name}' 失败: {str(e)}")
        
        if all_script_data:
            save_to_excel(all_script_data)
        
        # 构建显示内容
        display_text = []
        for script_data in all_script_data:
            for key, value in script_data.items():
                if key == "error":
                    display_text.append(f"错误: {value}")
                else:
                    display_text.append(f"{key.capitalize()}: {value}")
            display_text.append("-" * 50)
        
        return "处理成功", "\n".join(display_text)
    
    def refresh_file_list():
        script_files = get_script_files()
        if not script_files:
            return gr.Dropdown.update(choices=[], value=None), gr.Markdown("请上传txt剧本到/opt/Filmdataset/demo/juben_cn/batch_009")
        return gr.Dropdown.update(choices=script_files, value=script_files[0]), gr.Markdown("")
    
    script_files = get_script_files()
    
    with gr.Blocks(title="剧本信息提取系统") as interface:
        gr.Markdown("# 电影剧本信息提取系统")
        
        with gr.Row():
            with gr.Column(scale=1):
                process_btn = gr.Button("提取所有剧本信息", variant="primary")
                refresh_btn = gr.Button("🔄 刷新文件列表", variant="secondary")
            
            with gr.Column(scale=2):
                status_output = gr.Textbox(label="处理状态", interactive=False)
                info_output = gr.Textbox(label="提取结果", lines=15, interactive=False)
        
        process_btn.click(
            fn=process_scripts,
            inputs=[],
            outputs=[status_output, info_output]
        )
        
        refresh_btn.click(
            fn=refresh_file_list,
            inputs=[],
            outputs=[status_output]
        )
        
        if not script_files:
            gr.Markdown("提示：请将txt格式剧本上传到 `/opt/Filmdataset/demo/juben_cn/batch_009` 目录")
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7861)