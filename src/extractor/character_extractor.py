import os
import json
import gradio as gr
import pandas as pd
from typing import List, Dict, Any, Optional
import asyncio
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('/opt/rag_milvus_kb_project/.env')

# 从环境变量获取配置
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
API_VERSION = os.getenv('API_VERSION', '2024-02-15-preview')

if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT]):
    raise ValueError("Missing required environment variables: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT")

SCRIPT_DIR = "/opt/Filmdataset/demo/clean"
OUTPUT_DIR = "/opt/Filmdataset/demo/character"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 提示词定义（保持原格式）
PROMPT = """
- system提示词
    - 你是一位专业电影编剧。
 - 根据所给剧本，获取剧本名（script_name），总结该电影剧本的主要人物信息(包括主角、反派、重要支撑角色等对故事推动发挥重要作用的角色)，按如下内容输出。
        - 基本信息（basic_information）：包含姓名、性别、年龄、外貌特征、职业等信息；以一句话描述，字数50-100字。
        - 角色特征（characteristics）：性格特点、优点、弱点、技能、爱好、恐惧等信息；以一句话描述，字数50-100字。
        - 人物小传（biography）：角色动机、目标，以及他遇到的对抗性力量，他如何对抗冲突，最终的解决等。以及故事发展过程中，角色所经历的变化与发展轨迹(心理、情感以及行为的转变)。以一句话描述，字数50-100字。
        - 人物摘要（character_summary）:将basic_information、characteristics、biography三者结合，形成一个完整的角色摘要。以一句话描述，字数50-100字。
- 输出样例
    - [{"character_name":{
        "character_name":"XXX",
        "basic_information":"XXX",
        "characteristics":"XXX",
        "biography":"XXX",
        "character_summary":"XXX",
        "script_name":"XXX"
        },
        "character_name":{
        "character_name":"XXX",
        "basic_information":"XXX",
        "characteristics":"XXX",
        "biography":"XXX",
        "character_summary":"XXX",
        "script_name":"XXX"
        }
        ...,
        "character_name":{
        "character_name":"XXX",
        "basic_information":"XXX",
        "characteristics":"XXX",
        "biography":"XXX",
        "character_summary":"XXX",
        "script_name":"XXX"
        }
        ]
- 输出要求
   - 1、严格按照输出样例的JSON格式输出，不要新增或修改列表里json里的字段。
"""

# 模型调用函数
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
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": text}],
        temperature=1,
        top_p=0.7,
        response_format={"type": "json_object"}  # 强制要求JSON格式响应
    )
    return response.choices[0].message.content

# 人物信息提取函数
async def extract_character_information(prompt: str, script_path: str) -> Dict[str, Dict[str, Any]]:
    """提取单个剧本的人物信息"""
    try:
        with open(script_path, 'r', encoding='utf-8') as file:
            script_content = file.read()
        
        script_name = os.path.basename(script_path)
        result_text = await async_model_gpt4o_infer(prompt, script_content)
        
        # 解析JSON结果
        try:
            result_data = json.loads(result_text)
            characters = result_data if isinstance(result_data, list) else list(result_data.values())
        except json.JSONDecodeError as e:
            print(f"解析JSON失败: {e}，文件路径: {script_path}")
            return {
                "解析错误": {
                    "script_name": script_name,
                    "character_name": "解析错误",
                    "basic_information": f"JSON解析失败: {str(e)}",
                    "characteristics": "",
                    "character_summary": "",
                    "biography": ""
                }
            }
        
        # 处理人物重名
        character_dict = {}
        for char in characters:
            if 'character_name' not in char:
                continue
            char["script_name"] = script_name
            name = char["character_name"]
            if name in character_dict:
                count = 1
                while f"{name}_{count}" in character_dict:
                    count += 1
                name = f"{name}_{count}"
            character_dict[name] = char
        
        return character_dict

    except Exception as e:
        print(f"处理文件失败: {e}，文件路径: {script_path}")
        script_name = os.path.basename(script_path)
        return {
            "处理错误": {
                "script_name": script_name,
                "character_name": "处理错误",
                "basic_information": f"文件处理失败: {str(e)}",
                "characteristics": "",
                "character_summary": "",
                "biography": ""
            }
        }

# 保存到Excel函数
def save_to_excel(characters: Dict[str, Dict[str, Any]], 
                  output_path: str = os.path.join(OUTPUT_DIR, "character.xlsx")) -> None:
    """批量保存人物信息到Excel"""
    if not characters:
        print("没有需要保存的人物信息")
        return
    
    characters_list = list(characters.values())
    df = pd.DataFrame(characters_list)
    
    # 列名映射与排序
    column_mapping = {
        "script_name": "剧本名称",
        "character_name": "角色名称",
        "basic_information": "基本信息",
        "characteristics": "角色特征",
        "biography": "人物小传",
        "character_summary": "人物摘要"
    }
    df = df.rename(columns=column_mapping)
    df.insert(0, "序号", range(1, len(df) + 1))
    desired_columns = ["序号", "剧本名称", "角色名称", "基本信息", "角色特征", "人物小传", "人物摘要"]
    df = df[desired_columns]
    
    # 保存文件
    try:
        if os.path.exists(output_path):
            existing_df = pd.read_excel(output_path, engine='openpyxl')
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"成功追加数据，当前总记录数: {len(combined_df)}")
        else:
            df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"成功创建文件，初始记录数: {len(df)}")
    except Exception as e:
        print(f"保存Excel失败: {e}，尝试保存到临时文件")
        temp_path = f"{output_path}.temp.xlsx"
        df.to_excel(temp_path, index=False, engine='openpyxl')
        print(f"临时文件保存成功: {temp_path}")

# 批量处理入口函数
def process_batch_scripts():
    """处理目录下所有剧本文件"""
    script_files = [f for f in os.listdir(SCRIPT_DIR) 
                   if os.path.isfile(os.path.join(SCRIPT_DIR, f)) 
                   and f.lower().endswith('.txt')]  # 支持大小写敏感
    
    if not script_files:
        return "目录中没有找到txt剧本文件", []
    
    all_characters = {}
    error_count = 0
    
    for file in script_files:
        file_path = os.path.join(SCRIPT_DIR, file)
        try:
            chars = asyncio.run(extract_character_information(PROMPT, file_path))
            all_characters.update(chars)
        except Exception as e:
            error_count += 1
            print(f"文件 {file} 处理失败: {str(e)}")
    
    # 保存结果
    save_to_excel(all_characters)
    
    # 生成状态信息
    status = f"✅ 处理完成\n文件总数: {len(script_files)}\n成功提取角色数: {len(all_characters) - error_count}\n错误文件数: {error_count}"
    if error_count > 0:
        status += "\n⚠️ 错误详情请查看控制台日志"
    
    # 生成预览结果（限制2000字）
    preview = "\n".join([
        f"剧本: {char['script_name']}\n角色: {char['character_name']}\n摘要: {char['character_summary'][:200]}\n---"
        for char in list(all_characters.values())[:20]  # 最多显示前20个角色
    ])[:2000]  # 限制总长度
    
    return status, preview

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="批量剧本人物分析系统", css="style.css") as interface:
        gr.Markdown("# 📚 剧本人物信息批量提取系统")
        gr.Markdown("自动分析 `/opt/Filmdataset/demo/clean` 目录下的所有TXT剧本文件")
        
        with gr.Column(scale=1, min_width=600):
            status_box = gr.Textbox(
                label="处理状态", 
                lines=3, 
                interactive=False, 
                placeholder="等待处理..."
            )
            result_box = gr.Textbox(
                label="提取结果预览", 
                lines=10, 
                interactive=False, 
                placeholder="结果将显示在此处"
            )
            process_btn = gr.Button(
                "开始批量处理", 
                variant="primary", 
                size="lg", 
                icon="fa-solid fa-play"
            )
        
        # 绑定处理函数
        process_btn.click(
            fn=process_batch_scripts,
            outputs=[status_box, result_box]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_port=7860, share=False)