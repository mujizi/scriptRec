import os
import json
import asyncio
import re
import pandas as pd
from openai import AsyncAzureOpenAI
from typing import Dict, Any, List
from dotenv import load_dotenv
from datetime import datetime
import aiohttp
import sys
import logging
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm import async_model_infer
from extractor.single_script_processor import (
    split_text_by_5000_words,
    extract_scene_names,
    split_script_into_scenes,
    extract_tags_and_summaries,
    save_to_excel
)

# 创建必要的目录
base_dir = Path('/opt/rag_milvus_kb_project')
log_dir = base_dir / 'kb_data' / 'scene_result' / 'batch_000' / 'output_log'
log_dir.mkdir(parents=True, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv('/opt/rag_milvus_kb_project/.env')

# 从环境变量获取Azure OpenAI配置
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
API_VERSION = os.getenv('API_VERSION', '2024-02-15-preview')
AZURE_MODEL_NAME = os.getenv('AZURE_MODEL_NAME')

# 验证必要的环境变量
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
    # 强制输出到batch_000下的scene_name_json等目录
    base_output_dir = Path('/opt/rag_milvus_kb_project/kb_data/scene_result/batch_000')
    
    # 定义需要创建的目录
    directories = {
        'scene_summary_xlsx': base_output_dir / 'scene_summary_xlsx',
        'scene_summary_json': base_output_dir / 'scene_summary_json',
        'scene_name_json': base_output_dir / 'scene_name_json',
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
    failed_log_dir = Path('/opt/rag_milvus_kb_project/src/fail_process/batch_000')
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
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            script_text = file.read()
        
        # 1. 分割文本块
        text_blocks = split_text_by_5000_words(script_text)
        
        # 2. 提取场景名
        scene_result = await extract_scene_names(text_blocks, script_name)
        scene_names = scene_result["scenes"]
        
        if not scene_names:
            logger.error(f"No scene names extracted from {script_name}")
            return False
        
        # 保存场景名到JSON
        scene_name_file = os.path.join(
            output_dirs['scene_name_json'],
            f"{script_name}_scene_names.json"
        )
        with open(scene_name_file, "w", encoding="utf-8") as f:
            json.dump(scene_names, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved scene names to: {scene_name_file}")
        
        # 3. 分割场景
        scenes = split_script_into_scenes(script_text, scene_names, script_name)
        
        # 4. 提取标签和摘要
        tags_summaries = await extract_tags_and_summaries(scenes)
        
        # 保存摘要到JSON
        summary_file = os.path.join(
            output_dirs['scene_summary_json'],
            f"{script_name}_summaries.json"
        )
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(tags_summaries, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved summaries to: {summary_file}")
        
        # 5. 保存到Excel
        excel_file = os.path.join(
            output_dirs['scene_summary_xlsx'],
            f"{script_name}_scene_summaries.xlsx"
        )
        save_to_excel(scenes, tags_summaries, script_name, excel_file)
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
        # 指定处理fail_txt目录
        batch_dir = "/opt/rag_milvus_kb_project/kb_data/scene_result/batch_000/fail_txt"
        if not os.path.exists(batch_dir):
            raise FileNotFoundError(f"找不到目录: {batch_dir}")
            
        logger.info(f"Starting to process batch: {batch_dir}")
        await process_batch_directory(batch_dir)
            
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 