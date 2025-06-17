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

# 创建必要的目录
log_dir = Path('/opt/rag_milvus_kb_project/src/logs')
output_dir = Path('/opt/rag_milvus_kb_project/src/test_output')
log_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'script_processor.log'),
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

class ScriptProcessingError(Exception):
    """自定义异常类，用于处理脚本处理过程中的错误"""
    pass

async def async_model_gpt41_infer(instruct_text: str, raw_text: str) -> str:
    """
    使用Azure OpenAI API进行异步推理，包含错误处理和重试逻辑
    """
    text = f"{instruct_text} {raw_text}"
    logger.info(f"Processing text block of length: {len(raw_text)}")
    
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            client = AsyncAzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=API_VERSION
            )
            
            response = await client.chat.completions.create(
                model=AZURE_MODEL_NAME,
                messages=[
                    {"role": "user", "content": text},
                ],
                temperature=1,
                top_p=0.7,
            )
            return response.choices[0].message.content
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                logger.error(f"API调用最终失败: {str(e)}")
                raise ScriptProcessingError(f"API调用失败: {str(e)}")

def split_text_by_5000_words(text: str) -> Dict[int, str]:
    """
    将文本按句子分割成不超过5000字的块
    
    Args:
        text (str): 输入文本
    
    Returns:
        Dict[int, str]: 文本块字典，键为块索引，值为文本内容
    """
    # 使用正则表达式分割句子
    sentences = re.findall(r'.*?[？！。]|.*?(?=\n)|.*?(?=$)', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    chunks = {}
    current_chunk = []
    current_word_count = 0
    chunk_index = 0

    # 将句子组合成不超过5000字的块
    for sentence in sentences:
        current_word_count += len(sentence)
        current_chunk.append(sentence)
        if current_word_count >= 5000:
            chunks[chunk_index] = "\n".join(current_chunk)
            chunk_index += 1
            current_chunk = []
            current_word_count = 0

    # 处理最后一个块
    if current_chunk:
        chunks[chunk_index] = "\n".join(current_chunk)

    print(f"分割后的文本块数量: {len(chunks)}")
    return chunks

async def process_script(file_path: str) -> Dict[str, Any]:
    """
    处理单个剧本文件的主函数，包含完整的错误处理
    """
    try:
        # 获取剧本名称（不含扩展名）
        script_name = os.path.splitext(os.path.basename(file_path))[0]
        logger.info(f"开始处理剧本: {script_name}")
        
        # 读取剧本文件
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                script_text = file.read()
        except Exception as e:
            raise ScriptProcessingError(f"读取文件失败: {str(e)}")

        # 1. 分割文本块
        logger.info("正在分割文本...")
        text_blocks = split_text_by_5000_words(script_text)

        # 2. 提取场景名
        logger.info("正在提取场景名...")
        scene_result = await extract_scene_names(text_blocks, script_name)
        scene_names = scene_result["scenes"]

        if not scene_names:
            raise ScriptProcessingError("未能提取到任何场景名")

        # 3. 分割场景
        logger.info("正在分割场景...")
        scenes = split_script_into_scenes(script_text, scene_names, script_name)
        logger.info(f"成功分割 {len(scenes)} 个场景")

        # 4. 提取标签和摘要
        logger.info("正在提取标签和摘要...")
        tags_result = await extract_tags_and_summaries(scenes)
        tags_summaries = tags_result["tags_summaries"]

        # 5. 保存结果
        logger.info("正在保存结果...")
        excel_file = save_to_excel(scenes, tags_summaries, script_name)

        logger.info(f"剧本 {script_name} 处理完成")
        return {
            "status": "success",
            "message": f"成功处理文件 '{file_path}'，提取了 {len(scenes)} 个场景",
            "data": {
                "scene_names": scene_names,
                "scenes": scenes,
                "tags_summaries": tags_summaries,
                "excel_file": excel_file
            }
        }

    except ScriptProcessingError as e:
        logger.error(f"处理失败: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "data": None
        }
    except Exception as e:
        logger.error(f"未预期的错误: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"处理文件时发生未预期的错误: {str(e)}",
            "data": None
        }

async def extract_scene_names(text_blocks: Dict[int, str], script_name: str) -> Dict[str, Any]:
    """
    从文本块中提取场景名
    
    Args:
        text_blocks (Dict[int, str]): 文本块字典
        script_name (str): 剧本名称
    
    Returns:
        Dict[str, Any]: 提取的场景名和相关信息
    """
    prompt_template = '''
    你是一个场景名抽取专家。请根据输入的剧本，严格按照原文格式提取所有场景名。
    ## 背景：
    **场景名**：场景名以数字序号开头，序号后可能跟随"、"、"."或其他标点符号（严格按原文保留），后跟地点、时间等信息，元素顺序按原文呈现。
    ## 提取场景名的要求：
    - 必须完全按照剧本中出现的原始文本提取场景名（包括序号后的标点符号，如"1、"或"3."）
    - 抽取的文本必须是剧本中连续出现的完整场景名，不得修改任何字符
    - 序号为独立数字（如"3"和"4"是不同场景，禁止合并为"3.4."）
    - 跳过非场景名的陈述句
    - 必须提取所有场景名，包括带有"续"、"继续"、"待续"等字样的场景

    剧本：{input_text}

    输出结果：
    是一个合法的json，要求：
    - 键是场景名开头的纯数字序号（去除所有标点，仅保留数字，如"1、"对应键"1"）
    - 值是完整的原始场景名（包含序号后的所有原文标点和内容）
    - 顺序必须按数字序号从小到大排列
    - 不需要任何解释
    - 场景中的双引号请用反斜杠转义（如"对话：\"你好\""）'''

    results = await batch_process_blocks(text_blocks, prompt_template)
    all_scenes = {}
    gpt_responses = []

    for input_text, output_text in zip(results["inputs"], results["outputs"]):
        try:
            cleaned_output = re.sub(r'```json|```', '', output_text).strip()
            cleaned_output = cleaned_output.replace('\\"', '"')
            scenes = json.loads(cleaned_output)

            for key, value in scenes.items():
                if not key.isdigit():
                    continue
                all_scenes[key] = value

            gpt_responses.append(f"输入:\n{input_text}\n\n输出:\n{cleaned_output}\n\n")
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}\n原始响应内容:\n{output_text}")
            gpt_responses.append(f"输入:\n{input_text}\n\n输出(解析失败):\n{output_text}\n\n")

    sorted_scenes = dict(sorted(all_scenes.items(), key=lambda x: int(x[0])))
    
    scene_name_file = f"/opt/rag_milvus_kb_project/src/test_output/{script_name}_scene_names.json"
    with open(scene_name_file, "w", encoding="utf-8") as f:
        json.dump(sorted_scenes, f, ensure_ascii=False, indent=2)

    return {
        "scenes": sorted_scenes,
        "gpt_responses": gpt_responses,
        "scene_name_file": scene_name_file
    }

def split_script_into_scenes(script_text: str, scene_names: Dict[str, str], script_name: str) -> Dict[str, str]:
    """
    改进的场景分割函数，使用更强大的模式匹配和验证
    """
    scene_name_list = [v.strip() if v else '' for v in scene_names.values()]
    scene_name_set = set(scene_name_list)
    
    def build_scene_pattern(scene_name):
        # 处理特殊字符和格式
        name = scene_name.strip()
        # 转义特殊字符
        name = re.escape(name)
        # 处理可能的换行和空格变体
        name = name.replace(r'\n', r'[\n\s]*')
        name = name.replace(r'\s+', r'[\s\n]+')
        # 处理可能的标点符号变体
        name = name.replace(r'\.', r'[\.\。]')
        name = name.replace(r'\,', r'[\,\，]')
        return name

    patterns = [build_scene_pattern(name) for name in scene_name_list]
    big_pattern = '(' + '|'.join(patterns) + ')'
    
    # 打印每个场景名的匹配情况
    for name in scene_name_list:
        pattern = build_scene_pattern(name)
        matches = re.finditer(pattern, script_text)
        if matches:
            print(f"匹配到场景名: {repr(name)}")
            for match in matches:
                print(f"  位置: {match.start()}-{match.end()}")
        else:
            print(f"未匹配到场景名: {repr(name)}")
    
    parts = re.split(big_pattern, script_text)
    scenes_content = {}
    i = 1
    while i < len(parts) - 1:
        scene_name = parts[i].strip() if parts[i] else ''
        scene_content = parts[i+1].strip() if parts[i+1] else ''
        
        # 验证场景内容
        if scene_name in scene_name_set:
            # 检查场景内容是否有效
            if len(scene_content) >= 50:  # 设置最小内容长度
                scenes_content[scene_name] = scene_content
            else:
                print(f"警告: 场景 '{scene_name}' 的内容过短（{len(scene_content)}字符），已跳过")
        i += 2
    
    # 打印分割结果
    for k, v in scenes_content.items():
        print(f"分割到场景: {repr(k)}，内容长度: {len(v)}")
    
    return scenes_content

async def extract_tags_and_summaries(scenes: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """
    优化的场景摘要提取函数，使用批量处理
    """
    results = {}
    scene_items = list(scenes.items())
    batch_size = 5
    
    for i in range(0, len(scene_items), batch_size):
        batch = scene_items[i:i + batch_size]
        tasks = []
        
        for scene_name, scene_content in batch:
            prompt = f"""你是剧本场景提取摘要的专家，请为以下场景内容中精准提取核心摘要。请按以下要求处理输入内容：
    1.要求摘要表述宽泛，避免细节冗余
    2.摘要生成规则:
    2.1 内容要求：以 1-2 句话高度凝练场景，不出现具体人名，物品，仅保留关键信息
    2.2 语言规范：使用简洁书面语，避免标点冗余（仅保留必要逗号 / 句号），杜绝模糊词汇（如 "某种"" 相关 ""一系列"）
    2.3 输出格式：直接返回场景摘要文本
    输出例子：
    1.两个人进入了一个洞穴，发现了一些东西。
    2.一个人在古玩店介绍古董时，有人求鉴物品，另一个人发现物品内有重要信息。

    场景内容：
    {scene_content if scene_content else ''}

    请直接给出摘要，不需要其他格式。摘要应该清晰、准确地概括场景的主要内容。"""
            
            task = asyncio.create_task(async_model_infer(prompt))
            tasks.append((scene_name, task))
        
        # 等待所有任务完成
        for scene_name, task in tasks:
            try:
                response = await task
                summary = response.strip() if response else ''
                results[scene_name] = {
                    'summary': summary
                }
                print(f"场景 {scene_name} 摘要生成成功")
            except Exception as e:
                print(f"处理场景 {scene_name} 时出错: {str(e)}")
                results[scene_name] = {
                    'summary': ''
                }
        
        # 添加短暂延迟以避免API限制
        if i + batch_size < len(scene_items):
            await asyncio.sleep(0.5)
    
    return results

def save_to_excel(scenes: Dict[str, str], tags_summaries: Dict[str, Dict[str, str]], script_name: str, output_file: str = None) -> str:
    """
    将场景内容和摘要保存到Excel文件（即使没有有效内容也保存表头）
    """
    data = {
        "id": [],
        "场景名": [],
        "场景具体内容": [],
        "摘要": [],
        "剧本名": []
    }
    # 按场景编号排序
    sorted_scenes = sorted(scenes.items(), key=lambda x: int(re.search(r'^(\d+)', x[0]).group(1)) if re.search(r'^(\d+)', x[0]) else 0)
    current_id = 1
    for scene_name, scene_content in sorted_scenes:
        if scene_content is None:
            scene_content = ''
        summary_info = tags_summaries.get(scene_name, {})
        summary = summary_info.get('summary', '')
        # 临时注释掉内容长度过滤，便于调试
        # if len(scene_content.strip()) < 30:
        #     print(f"警告: 场景 '{scene_name}' 的内容过短（{len(scene_content.strip())}字），已跳过")
        #     continue
        data["id"].append(current_id)
        data["场景名"].append(scene_name)
        data["场景具体内容"].append(scene_content)
        data["摘要"].append(summary)
        data["剧本名"].append(script_name)
        current_id += 1
    df = pd.DataFrame(data)
    df = df[['id', '场景名', '场景具体内容', '摘要', '剧本名']]
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/opt/rag_milvus_kb_project/src/test_output/{script_name}_scene_summaries_{timestamp}.xlsx"
    try:
        # 即使df为空也保存表头
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"场景内容和摘要已保存到: {output_file}")
        print(f"共保存了 {len(df)} 个有效场景（已过滤掉内容过短的场景）")
    except Exception as e:
        print(f"保存Excel时出错: {e}")
    return output_file

async def batch_process_blocks(blocks: Dict[int, str], prompt: str) -> Dict[str, List[str]]:
    """
    优化的批量处理文本块函数，使用更高效的并发处理
    """
    MAX_CONCURRENT_REQUESTS = 20  # 增加并发数
    BATCH_SIZE = 5  # 每批处理的块数
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def process_block(block: str) -> Dict[str, str]:
        async with semaphore:
            try:
                result = await async_model_gpt41_infer(prompt, block)
                return {
                    "input": block,
                    "output": result,
                    "status": "success"
                }
            except Exception as e:
                print(f"处理块时出错: {str(e)}")
                return {
                    "input": block,
                    "output": "",
                    "status": "error",
                    "error": str(e)
                }

    # 将块分批处理
    block_values = list(blocks.values())
    results = []
    
    for i in range(0, len(block_values), BATCH_SIZE):
        batch = block_values[i:i + BATCH_SIZE]
        tasks = [process_block(block) for block in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        
        # 添加短暂延迟以避免API限制
        if i + BATCH_SIZE < len(block_values):
            await asyncio.sleep(0.5)

    # 处理结果
    successful_results = [r for r in results if r["status"] == "success"]
    failed_results = [r for r in results if r["status"] == "error"]
    
    if failed_results:
        print(f"警告: {len(failed_results)} 个块处理失败")
        for failed in failed_results:
            print(f"失败块: {failed['error']}")

    return {
        "inputs": [r["input"] for r in successful_results],
        "outputs": [r["output"] for r in successful_results],
        "failed_count": len(failed_results)
    }

async def main():
    """
    主函数
    """
    try:
        # 获取脚本文件路径
        script_file = "/opt/rag_milvus_kb_project/kb_data/test/1976年魁北克党选举.txt"
        if not os.path.exists(script_file):
            raise FileNotFoundError(f"找不到脚本文件: {script_file}")
        
        # 读取脚本内容
        with open(script_file, "r", encoding="utf-8") as f:
            script_content = f.read()
        
        # 获取剧本名称
        script_name = os.path.splitext(os.path.basename(script_file))[0]
        
        # 分割文本块
        print("正在分割文本...")
        text_blocks = split_text_by_5000_words(script_content)
        
        # 提取场景名
        print("正在提取场景名...")
        scene_result = await extract_scene_names(text_blocks, script_name)
        scene_names = scene_result["scenes"]
        
        if not scene_names:
            raise ValueError("未能提取到任何场景名")
        
        # 分割场景
        print("正在分割场景...")
        scenes = split_script_into_scenes(script_content, scene_names, script_name)
        print(f"成功分割 {len(scenes)} 个场景")
        
        # 提取摘要
        print("正在提取摘要...")
        tags_summaries = await extract_tags_and_summaries(scenes)
        
        # 保存到Excel
        print("正在保存到Excel...")
        excel_file = save_to_excel(scenes, tags_summaries, script_name)
        print(f"已保存到Excel文件: {excel_file}")
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 