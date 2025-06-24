import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import asyncio
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm import async_model_infer

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='批量提取剧本人物信息')
    parser.add_argument('--batch', type=str, default='batch_006', 
                       help='要处理的batch目录名称 (例如: batch_006, batch_007)')
    return parser.parse_args()

# 解析命令行参数
args = parse_arguments()
batch_name = args.batch

# 创建必要的目录
base_dir = Path('/opt/rag_milvus_kb_project')
log_dir = base_dir / 'kb_data' / 'character_result' / batch_name / 'output_log'
log_dir.mkdir(parents=True, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'character_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv('/opt/rag_milvus_kb_project/.env')

# 从环境变量获取配置
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
API_VERSION = os.getenv('API_VERSION', '2024-02-15-preview')

if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT]):
    raise ValueError("Missing required environment variables: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT")

def create_output_directories(batch_name: str) -> Dict[str, str]:
    """
    为每个batch创建必要的输出目录
    
    Args:
        batch_name (str): batch目录名称
    
    Returns:
        Dict[str, str]: 包含各个输出目录路径的字典
    """
    # 动态输出到指定batch下的目录
    base_output_dir = Path(f'/opt/rag_milvus_kb_project/kb_data/character_result/{batch_name}')
    
    # 定义需要创建的目录
    directories = {
        'character_xlsx': base_output_dir / 'character_xlsx',
        'output_log': base_output_dir / 'output_log'
    }
    
    # 创建目录
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return {k: str(v) for k, v in directories.items()}

def save_failed_files(batch_name: str, failed_files: List[str], error_messages: Dict[str, str]):
    """
    将处理失败的文件记录到专门的日志文件中
    
    Args:
        batch_name (str): batch目录名称
        failed_files (List[str]): 失败文件列表
        error_messages (Dict[str, str]): 错误信息字典，key为文件名，value为错误信息
    """
    failed_log_dir = Path(f'/opt/rag_milvus_kb_project/src/fail_process/character_{batch_name}')
    failed_log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failed_log_file = failed_log_dir / f"failed_files_{batch_name}_{timestamp}.log"
    
    with open(failed_log_file, 'w', encoding='utf-8') as f:
        f.write(f"Batch: {batch_name}\n")
        f.write(f"Processing Time: {timestamp}\n")
        f.write(f"Total Failed Files: {len(failed_files)}\n")
        f.write("\nFailed Files Details:\n")
        f.write("-" * 50 + "\n")
        
        for file_name in failed_files:
            error_msg = error_messages.get(file_name, "Unknown error")
            f.write(f"File: {file_name}\n")
            f.write(f"Error: {error_msg}\n")
            f.write("-" * 50 + "\n")
    
    logger.info(f"Failed files log saved to: {failed_log_file}")

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
async def async_model_gpt41_infer(instruct_text: str, raw_text: str) -> str:
    """调用Azure OpenAI的gpt-4o-mini模型进行推理，确保返回JSON格式"""
    text = f"{instruct_text} {raw_text}"
    logger.info(f"Processing text block of length: {len(raw_text)}")
    client = AsyncAzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=API_VERSION
    )
    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
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
        result_text = await async_model_gpt41_infer(prompt, script_content)
        
        # 解析JSON结果
        try:
            result_data = json.loads(result_text)
            characters = result_data if isinstance(result_data, list) else list(result_data.values())
        except json.JSONDecodeError as e:
            logger.error(f"解析JSON失败: {e}，文件路径: {script_path}")
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
        logger.error(f"处理文件失败: {e}，文件路径: {script_path}")
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
                  output_path: str) -> None:
    """批量保存人物信息到Excel"""
    if not characters:
        logger.warning("没有需要保存的人物信息")
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
            logger.info(f"成功追加数据，当前总记录数: {len(combined_df)}")
        else:
            df.to_excel(output_path, index=False, engine='openpyxl')
            logger.info(f"成功创建文件，初始记录数: {len(df)}")
    except Exception as e:
        logger.error(f"保存Excel失败: {e}，尝试保存到临时文件")
        temp_path = f"{output_path}.temp.xlsx"
        df.to_excel(temp_path, index=False, engine='openpyxl')
        logger.info(f"临时文件保存成功: {temp_path}")

async def process_single_file(
    file_path: str,
    output_dirs: Dict[str, str],
    batch_name: str
) -> bool:
    """
    处理单个文件
    
    Args:
        file_path (str): 文件路径
        output_dirs (Dict[str, str]): 输出目录字典
        batch_name (str): batch目录名称
    
    Returns:
        bool: 处理是否成功
    """
    try:
        script_name = os.path.splitext(os.path.basename(file_path))[0]
        logger.info(f"Processing file: {script_name}")
        
        # 提取人物信息
        characters = await extract_character_information(PROMPT, file_path)
        
        if not characters:
            logger.error(f"No characters extracted from {script_name}")
            return False
        
        # 保存到Excel
        excel_file = os.path.join(
            output_dirs['character_xlsx'],
            f"{script_name}_characters.xlsx"
        )
        save_to_excel(characters, excel_file)
        logger.info(f"Saved Excel to: {excel_file}")
        
        logger.info(f"Successfully processed {script_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {script_name}: {str(e)}")
        return False

async def process_batch_directory(batch_dir: str):
    """
    处理整个batch目录
    
    Args:
        batch_dir (str): batch目录路径
    """
    batch_name = os.path.basename(batch_dir)
    logger.info(f"Processing batch directory: {batch_name}")
    
    # 创建输出目录
    output_dirs = create_output_directories(batch_name)
    
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(batch_dir) if f.endswith('.txt')]
    total_files = len(txt_files)
    processed_files = 0
    failed_files = []
    error_messages = {}
    
    # 处理每个文件
    for txt_file in txt_files:
        file_path = os.path.join(batch_dir, txt_file)
        try:
            success = await process_single_file(file_path, output_dirs, batch_name)
            
            if success:
                processed_files += 1
            else:
                failed_files.append(txt_file)
                error_messages[txt_file] = "Processing failed"
        except Exception as e:
            failed_files.append(txt_file)
            error_messages[txt_file] = str(e)
        
        logger.info(f"Progress: {processed_files}/{total_files} files processed")
    
    # 记录处理结果
    logger.info(f"Batch {batch_name} processing completed:")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Successfully processed: {processed_files}")
    logger.info(f"Failed files: {len(failed_files)}")
    
    # 保存失败文件记录
    if failed_files:
        save_failed_files(batch_name, failed_files, error_messages)
        logger.info("Failed files list:")
        for file in failed_files:
            logger.info(f"- {file}")

async def main():
    """
    主函数
    """
    try:
        # 动态指定处理目录
        batch_dir = f"/opt/rag_milvus_kb_project/kb_data/script/juben_cn/{batch_name}"
        if not os.path.exists(batch_dir):
            raise FileNotFoundError(f"找不到目录: {batch_dir}")
            
        logger.info(f"Starting to process batch: {batch_dir}")
        await process_batch_directory(batch_dir)
            
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())