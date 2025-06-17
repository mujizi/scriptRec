import gradio as gr
import os
import json
import asyncio
from openai import AsyncAzureOpenAI
import pandas as pd
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('/opt/rag_milvus_kb_project/.env')

# 从环境变量获取配置
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
API_VERSION = os.getenv('API_VERSION', '2024-02-15-preview')
AZURE_MODEL_NAME = os.getenv('AZURE_MODEL_NAME')
if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT]):
    raise ValueError("Missing required environment variables: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT")

# 最大并发请求数
MAX_CONCURRENT_REQUESTS = 10

# 创建必要的目录
os.makedirs("/opt/Filmdataset/demo/output", exist_ok=True)

async def async_model_gpt41_infer(instruct_text: str, raw_text: str) -> str:
    text = f"{instruct_text} {raw_text}"
    print(f"Processing text block of length: {len(raw_text)}")
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

def split_text_by_5000_words(text: str) -> Dict[int, str]:
    sentences = re.findall(r'.*?[？！。]|.*?(?=\n)|.*?(?=$)', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    chunks = {}
    current_chunk = []
    current_word_count = 0
    chunk_index = 0

    for sentence in sentences:
        current_word_count += len(sentence)
        current_chunk.append(sentence)
        if current_word_count >= 5000:
            chunks[chunk_index] = "\n".join(current_chunk)
            chunk_index += 1
            current_chunk = []
            current_word_count = 0

    if current_chunk:
        chunks[chunk_index] = "\n".join(current_chunk)

    print(f"分割后的文本块数量: {len(chunks)}")
    return chunks

async def batch_process_blocks(blocks: Dict[int, str], prompt: str, progress: gr.Progress) -> Dict[str, List[str]]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def process_block(block: str) -> Dict[str, str]:
        async with semaphore:
            result = await async_model_gpt41_infer(prompt, block)
            return {
                "input": block[:500] + "..." if len(block) > 500 else block,
                "output": result
            }

    tasks = []
    total_blocks = len(blocks)

    for i, block in blocks.items():
        task = asyncio.create_task(process_block(block))
        tasks.append(task)

        # 更新进度
        if i % 5 == 0 or i == total_blocks - 1:
            progress(i / total_blocks, desc=f"正在处理文本块 {i + 1}/{total_blocks}")
            await asyncio.sleep(0.1)

    results = await asyncio.gather(*tasks)

    return {
        "inputs": [r["input"] for r in results],
        "outputs": [r["output"] for r in results]
    }

async def extract_scene_names(text_blocks: Dict[int, str], progress: gr.Progress, script_name: str) -> Dict[str, Any]:
    prompt_template =     '''你是一个场景名抽取专家。请根据输入的剧本，严格按照原文格式提取所有场景名。
## 背景：
**场景名**：场景名以数字序号开头，序号后可能跟随"、"、"."或其他标点符号（严格按原文保留），后跟地点、时间等信息，元素顺序按原文呈现。
## 提取场景名的要求：
 - 必须完全按照剧本中出现的原始文本提取场景名（包括序号后的标点符号，如"1、"或"3."）
 - 抽取的文本必须是剧本中连续出现的完整场景名，不得修改任何字符
 - 序号为独立数字（如"3"和"4"是不同场景，禁止合并为"3.4."）
 - 跳过非场景名的陈述句

剧本：{input_text}

输出结果：
是一个合法的json，要求：
- 键是场景名开头的纯数字序号（去除所有标点，仅保留数字，如"1、"对应键"1"）
- 值是完整的原始场景名（包含序号后的所有原文标点和内容）
- 顺序必须按数字序号从小到大排列
- 不需要任何解释
- 场景中的双引号请用反斜杠转义（如"对话：\"你好\""）'''

    progress(0, desc="开始提取场景名...")
    results = await batch_process_blocks(text_blocks, prompt_template, progress)

    all_scenes = {}
    gpt_responses = []

    for input_text, output_text in zip(results["inputs"], results["outputs"]):
        try:
            # 清理Markdown标记并处理转义引号
            cleaned_output = re.sub(r'```json|```', '', output_text).strip()
            # 修复可能的多余转义（如果模型错误生成双重转义）
            cleaned_output = cleaned_output.replace('\\"', '"')
            scenes = json.loads(cleaned_output)

            # 验证序号合法性（确保键为纯数字）
            for key, value in scenes.items():
                if not key.isdigit():
                    continue  # 过滤非数字键
                all_scenes[key] = value  # 直接使用原始场景名

            gpt_responses.append(f"输入:\n{input_text}\n\n输出:\n{cleaned_output}\n\n")
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}\n原始响应内容:\n{output_text}")
            gpt_responses.append(f"输入:\n{input_text}\n\n输出(解析失败):\n{output_text}\n\n")

    # 按数字序号排序
    sorted_scenes = dict(sorted(all_scenes.items(), key=lambda x: int(x[0])))

    # 保存时确保原始符号保留（JSON自动处理转义）
    scene_name_file = f"/opt/rag_milvus_kb_project/src/test_output/{script_name}_scene_names.json"
    with open(scene_name_file, "w", encoding="utf-8") as f:
        json.dump(sorted_scenes, f, ensure_ascii=False, indent=2, separators=(',', ': '))

    return {
        "scenes": sorted_scenes,
        "gpt_responses": gpt_responses,
        "scene_name_file": scene_name_file
    }
    
def split_script_into_scenes(script_text: str, scene_names: Dict[str, str], script_name: str) -> Dict[str, str]:
    scenes_content = {}
    start = 0

    # 按场景名顺序分割剧本
    sorted_scenes = sorted(scene_names.items(), key=lambda x: int(x[0]))

    for i, (scene_num, scene_name) in enumerate(sorted_scenes):
        try:
            # 使用正则表达式匹配场景名，忽略多个空格
            pattern = re.compile(r'\s+'.join(re.escape(part) for part in scene_name.split()))
            match = pattern.search(script_text, start)
            if not match:
                continue
            end = match.start()

            # 保存上一个场景的内容
            if start < end and i > 0:  # Skip the first chunk as it is the title
                previous_scene_name = sorted_scenes[i - 1][1]
                scenes_content[previous_scene_name] = script_text[start:end].strip()

            start = end
        except Exception as e:
            print(f"Error splitting script: {e}")

    # 添加最后一个场景
    if start < len(script_text):
        last_scene_name = sorted_scenes[-1][1]
        scenes_content[last_scene_name] = script_text[start:].strip()

    # 保存场景内容到JSON文件
    scene_content_file = f"/opt/rag_milvus_kb_project/src/test_output/{script_name}_scenes.json"
    with open(scene_content_file, "w", encoding="utf-8") as f:
        json.dump(scenes_content, f, ensure_ascii=False, indent=2)
    return scenes_content

async def extract_tags_and_summaries(scenes: Dict[str, str], progress: gr.Progress) -> Dict[str, Any]:
    prompt_template = '''
    你是剧本场景提取摘要的专家，需从剧本场景中精准提取核心摘要。请按以下要求处理输入内容：
    1.要求摘要表述宽泛，避免细节冗余
    2.摘要生成规则:
    2.1 内容要求：以 1-2 句话高度凝练场景，不出现具体人名，物品，仅保留关键信息
    2.2 语言规范：使用简洁书面语，避免标点冗余（仅保留必要逗号 / 句号），杜绝模糊词汇（如 "某种"" 相关 ""一系列"）
    2.3 输出格式：直接返回场景摘要文本
    输出例子：
    1.两个人进入了一个洞穴，发现了一些东西。
    2.一个人在古玩店介绍古董时，有人求鉴物品，另一个人发现物品内有重要信息。
    '''
    print('extract_tags_and_summaries')
    progress(0, desc="开始提取摘要...")
    blocks = {i: content for i, content in enumerate(scenes.values())}
    print('blocks:', blocks)
    results = await batch_process_blocks(blocks, prompt_template, progress)

    tags_summaries = {}
    gpt_responses = []
    scene_names = list(scenes.keys())
    for idx, (input_text, output_text) in enumerate(zip(results["inputs"], results["outputs"])):
        scene_name = scene_names[idx]
        # 清理可能的Markdown格式（如果有的话）
        cleaned_output = re.sub(r'```.*?```', '', output_text, flags=re.DOTALL).strip()
        # 修改这里，将idx和结果存到一个字典中
        tags_summaries[scene_name] = {
            "idx": idx + 1,  # 假设idx从1开始
            "summary": cleaned_output
        }
        gpt_responses.append(f"场景 '{scene_name}':\n输入:\n{input_text}\n\n输出:\n{output_text}\n\n")
    return {
        "tags_summaries": tags_summaries,
        "gpt_responses": gpt_responses
    }

def save_to_excel(tags_summaries: Dict[str, Dict[str, Any]], scenes: Dict[str, str], script_name: str) -> str:
    data = {
        "id": [],
        "场景名": [],
        "场景具体内容": [],
        "摘要": [],
        "剧本名": []
    }

    for scene_name, info in tags_summaries.items():
        idx = info["idx"]
        summary = info["summary"]
        data["id"].append(idx)
        data["场景名"].append(scene_name)
        data["场景具体内容"].append(scenes.get(scene_name, ""))
        data["摘要"].append(summary)
        data["剧本名"].append(script_name)

    df = pd.DataFrame(data)
    output_path = f"/opt/rag_milvus_kb_project/src/test_output/{script_name}_scene_summaries.xlsx"
    df.to_excel(output_path, index=False)
    print(f"结果已保存到: {output_path}")
    return output_path

async def process_script(file_path: str, progress: gr.Progress) -> Dict[str, Any]:
    try:
        gpt_results = []
        script_name = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, 'r', encoding='utf-8') as file:
            script_text = file.read()

        # 1. 分割文本块
        progress(0.1, desc="正在分割文本...")
        text_blocks = split_text_by_5000_words(script_text)

        split_blocks_display = "\n\n".join(
            [f"文本块 {i} (长度: {len(block)}字符):\n{block[:200]}..." if len(block) > 200 else block
             for i, block in text_blocks.items()]
        )

        # 2. 提取场景名
        progress(0.2, desc="准备提取场景名...")
        scene_result = await extract_scene_names(text_blocks, progress, script_name)
        scene_names = scene_result["scenes"]
        gpt_results.extend(scene_result["gpt_responses"])

        if not scene_names:
            return {
                "result": f"错误: 未能提取到任何场景名",
                "split_blocks": "未能成功分割文本块",
                "scenes": "未提取到场景",
                "gpt_outputs": "\n".join(gpt_results),
                "scene_name_file": None,
                "scene_content_file": None,
                "excel_file": None,
                "excel_df": None
            }

        # 3. 分割场景
        progress(0.7, desc="正在分割场景...")
        scenes = split_script_into_scenes(script_text, scene_names, script_name)

        # 4. 提取标签和摘要
        progress(0.8, desc="准备提取标签和摘要...")
        tags_result = await extract_tags_and_summaries(scenes, progress)
        tags_summaries = tags_result["tags_summaries"]
        gpt_results.extend(tags_result["gpt_responses"])

        # 5. 保存结果
        progress(0.95, desc="正在保存结果...")
        excel_file = save_to_excel(tags_summaries, scenes, script_name)
        excel_df = pd.read_excel(excel_file)

        progress(1.0, desc="处理完成！")

        scenes_display = "\n\n".join(
            [f"场景 '{name}' (长度: {len(scene)}字符):\n{scene[:200]}..." if len(scene) > 200 else scene
             for name, scene in scenes.items()]
        )

        gpt_output_display = "=== GPT处理结果 ===\n\n" + "\n".join(gpt_results)

        scene_content_file = f"/opt/Filmdataset/demo/output/{script_name}_scenes.json"

        return {
            "result": f"成功处理文件 '{file_path}'，提取了 {len(scenes)} 个场景",
            "split_blocks": split_blocks_display,
            "scenes": scenes_display,
            "gpt_outputs": gpt_output_display,
            "scene_name_file": scene_result["scene_name_file"],
            "scene_content_file": scene_content_file,
            "excel_file": excel_file,
            "excel_df": excel_df
        }
    except Exception as e:
        return {
            "result": f"处理文件 '{file_path}' 时出错: {str(e)}",
            "split_blocks": f"分割文本块时出错: {str(e)}",
            "scenes": f"提取场景时出错: {str(e)}",
            "gpt_outputs": f"GPT处理过程中出错: {str(e)}",
            "scene_name_file": None,
            "scene_content_file": None,
            "excel_file": None,
            "excel_df": None
        }

def main():
    current_dir = "/opt/rag_milvus_kb_project/kb_data/test"
    txt_files = [f for f in os.listdir(current_dir) if f.endswith('.txt')]

    if not txt_files:
        print("当前目录下没有找到 .txt 文件。")
        return

    progress = gr.Progress()
    failed_files = []

    for file in txt_files[:100]:
        file_path = os.path.join(current_dir, file)
        result = asyncio.run(process_script(file_path, progress))
        if "错误" in result["result"]:
            print(result["result"])
            failed_files.append(file_path)
        else:
            print(result["result"])

    if failed_files:
        print("以下文件处理失败：")
        for file in failed_files:
            print(file)

if __name__ == "__main__":
    main()