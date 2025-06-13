import json
import os
import asyncio
from typing import List, Dict
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('/opt/rag_milvus_kb_project/.env')

# 从环境变量获取配置
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
API_VERSION = os.getenv('API_VERSION', '2025-01-01-preview')

if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT]):
    raise ValueError("Missing required environment variables: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT")

def split_text_into_chunks(lines: List[str], chunk_size: int = 100) -> List[dict]:
    """
    将文本行列表分割成带有原始行号的块。

    Args:
        lines (List[str]): 文件的所有行。
        chunk_size (int): 每个块的最大行数。

    Returns:
        List[dict]: 一个字典列表，每个字典包含 'start_line' 和 'content'。
    """
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i:i + chunk_size]
        chunk_content = "".join(chunk_lines)
        # +1 因为行号是从1开始的
        start_line = i + 1
        chunks.append({"start_line": start_line, "content": chunk_content})
    return chunks

def get_analysis_prompt(script_chunk: str, start_line: int, total_lines: int) -> str:
    """
    生成用于剧本分析的提示。
    """
    # 场景名格式参考
    scene_format_examples = """
    参考标准的场景名格式：
    1 广西十万大山山脉 日 外
    2.内景，冷饮店，晚上
    43 森林深处 外 昏
    10.外景，宗教中心附近，接前景
    内景，保罗的车里，腐殖质土处理场，白天
    69.交叉剪辑，酒吧里，接前景
    9.内景，洛杉矶商务午餐馆，中午
    """

    prompt = f"""
你是一位专业的剧本分析师。你的任务是通读给定的剧本文本，识别出所有可能的不规范场景名。

**任务要求:**
1.  **分析文本**: 下面是一段剧本，行号从 {start_line} 开始。整个剧本总共有 {total_lines} 行。
2.  **识别场景名**: 找出所有符合或不符合规范的场景名。
3.  **修正不规范场景名**: 如果场景名不规范（例如，缺少序号、格式混乱等），请根据场景内容和上下文尽可能地完善它。特别注意为缺少序号的场景名补上正确的序号。
4.  **提供位置**: 给出修正后场景名所在的 **原始准确行号**。
5.  **输出格式**: 将所有需要修改的行（即，所有不规范并被你修正过的场景名）以JSON对象格式返回。键是 **原始行号** (字符串形式)，值是 **修正后的完整场景名**。

{scene_format_examples}

**约束:**
-   只返回包含待修改内容的JSON对象。
-   如果分析下来，文本块中没有任何需要修改的场景名，请返回一个空的JSON对象 `{{}}`。
-   确保返回的JSON格式严格正确。
-   行号必须是相对于整个文件的准确行号。

**待分析的剧本文本块 (从第 {start_line} 行开始):**
---
{script_chunk}
---
"""
    return prompt

async def analyze_script_chunk(script_chunk: str, start_line: int, total_lines: int) -> Dict[str, str]:
    """
    (异步真实函数) 分析剧本块并返回待修改的行。
    """
    prompt = get_analysis_prompt(script_chunk, start_line, total_lines)
    
    try:
        client = AsyncAzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,  
            api_version=API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        
        print(f"正在向Azure OpenAI API发送从第 {start_line} 行开始的文本块进行分析...")
        
        response = await client.chat.completions.create(
            model="gpt-4.1", # 您的模型部署名
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.5, # 降低温度以获得更确定的、格式化的输出
        )
        
        raw_json = response.choices[0].message.content
        print(f"从第 {start_line} 行开始的文本块分析完成。")
        return json.loads(raw_json)

    except json.JSONDecodeError:
        print(f"错误: API为第 {start_line} 行开始的块返回的不是有效的JSON。")
        return {}
    except Exception as e:
        print(f"调用Azure OpenAI API时发生错误 (起始行 {start_line}): {e}")
        return {}

async def analyze_screenplay(file_path: str, chunk_size: int = 100) -> Dict[str, str]:
    """
    (异步) 分割、分析整个剧本文件，并返回一个包含所有待修改内容的JSON。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return {}

    total_lines = len(lines)
    chunks = split_text_into_chunks(lines, chunk_size)
    all_modifications = {}

    # 创建一组并发任务
    tasks = [
        analyze_script_chunk(
            chunk['content'],
            chunk['start_line'],
            total_lines
        ) for chunk in chunks
    ]
    
    # 并发执行所有分析任务
    modification_results = await asyncio.gather(*tasks)

    # 收集所有结果
    for modifications in modification_results:
        all_modifications.update(modifications)

    # 按行号排序
    sorted_modifications = dict(sorted(all_modifications.items(), key=lambda item: int(item[0])))
    
    # --- 保存生成的JSON ---
    try:
        # 获取项目根目录，以便构建输出路径
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_dir = os.path.join(project_root, "src", "test_output")
        os.makedirs(output_dir, exist_ok=True)

        # 从输入文件路径派生输出文件名
        base_filename = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(base_filename)[0]
        output_filename = f"{filename_without_ext}_modifications.json"
        output_filepath = os.path.join(output_dir, output_filename)

        # 写入JSON文件
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(sorted_modifications, f, ensure_ascii=False, indent=4)
        print(f"成功将修改建议保存到: {output_filepath}")
    except Exception as e:
        print(f"错误: 保存JSON文件失败: {e}")
    # --- 保存结束 ---

    return sorted_modifications 