import os
import json
import gradio as gr
from pathlib import Path
import shutil
from typing import List, Dict, Optional

class MovieScriptRenamer:
    def __init__(self, base_juben_cn_dir: str, base_json_dir: str):
        """
        初始化电影剧本重命名器
        
        Args:
            base_juben_cn_dir: 中文剧本文件的基础目录
            base_json_dir: JSON映射文件的基础目录
        """
        self.base_juben_cn_dir = Path(base_juben_cn_dir)
        self.base_json_dir = Path(base_json_dir)
        
        # 确保目录存在
        if not self.base_juben_cn_dir.exists():
            raise FileNotFoundError(f"剧本目录不存在: {base_juben_cn_dir}")
        if not self.base_json_dir.exists():
            raise FileNotFoundError(f"JSON目录不存在: {base_json_dir}")
    
    def get_subdirectories(self) -> List[str]:
        """获取所有可用的子目录"""
        return sorted([str(dir.name) for dir in self.base_juben_cn_dir.iterdir() if dir.is_dir()])
    
    def get_mapping_file(self, subdirectory: str) -> Optional[Path]:
        """获取指定子目录对应的映射文件"""
        json_file = self.base_json_dir / f"{subdirectory}_results.json"
        return json_file if json_file.exists() else None
    
    def load_mapping(self, json_file: Path) -> Dict[str, str]:
        """加载JSON映射文件"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载JSON文件失败: {e}")
            return {}
    
    def preview_rename(self, subdirectory: str) -> List[Dict[str, str]]:
        """预览重命名操作"""
        results = []
        
        # 获取映射文件
        mapping_file = self.get_mapping_file(subdirectory)
        if not mapping_file:
            return [{"状态": "错误", "详情": f"找不到映射文件: {subdirectory}_results.json"}]
        
        # 加载映射
        mapping = self.load_mapping(mapping_file)
        if not mapping:
            return [{"状态": "错误", "详情": f"映射文件为空或格式错误: {mapping_file}"}]
        
        # 获取目标目录
        target_dir = self.base_juben_cn_dir / subdirectory
        if not target_dir.exists():
            return [{"状态": "错误", "详情": f"目标目录不存在: {target_dir}"}]
        
        # 遍历目录中的文件，生成预览
        for file in target_dir.iterdir():
            if file.is_file():
                # 提取文件名（不包含扩展名）
                base_name = file.stem
                ext = file.suffix
                
                # 检查是否在映射中
                if base_name in mapping:
                    new_name = mapping[base_name] + ext
                    results.append({
                        "原文件名": file.name,
                        "新文件名": new_name,
                        "状态": "将重命名"
                    })
                else:
                    results.append({
                        "原文件名": file.name,
                        "新文件名": file.name,
                        "状态": "未找到映射"
                    })
        
        return results
    
    def perform_rename(self, subdirectory: str) -> List[Dict[str, str]]:
        """执行重命名操作"""
        results = []
        
        # 获取映射文件
        mapping_file = self.get_mapping_file(subdirectory)
        if not mapping_file:
            return [{"状态": "错误", "详情": f"找不到映射文件: {subdirectory}_results.json"}]
        
        # 加载映射
        mapping = self.load_mapping(mapping_file)
        if not mapping:
            return [{"状态": "错误", "详情": f"映射文件为空或格式错误: {mapping_file}"}]
        
        # 获取目标目录
        target_dir = self.base_juben_cn_dir / subdirectory
        if not target_dir.exists():
            return [{"状态": "错误", "详情": f"目标目录不存在: {target_dir}"}]
        
        # 遍历目录中的文件，执行重命名
        for file in target_dir.iterdir():
            if file.is_file():
                # 提取文件名（不包含扩展名）
                base_name = file.stem
                ext = file.suffix
                
                # 检查是否在映射中
                if base_name in mapping:
                    new_name = mapping[base_name] + ext
                    new_path = file.parent / new_name
                    
                    try:
                        # 执行重命名
                        shutil.move(file, new_path)
                        results.append({
                            "原文件名": file.name,
                            "新文件名": new_name,
                            "状态": "已重命名"
                        })
                    except Exception as e:
                        results.append({
                            "原文件名": file.name,
                            "新文件名": new_name,
                            "状态": f"错误: {str(e)}"
                        })
                else:
                    results.append({
                        "原文件名": file.name,
                        "新文件名": file.name,
                        "状态": "未找到映射"
                    })
        
        return results

def create_interface():
    """创建Gradio界面"""
    base_juben_cn_dir = "/opt/Filmdataset/demo/juben_cn"
    base_json_dir = "/opt/Filmdataset/demo/juben_json"
    
    try:
        renamer = MovieScriptRenamer(base_juben_cn_dir, base_json_dir)
    except Exception as e:
        return gr.Interface(
            fn=lambda x: f"初始化失败: {str(e)}",
            inputs=gr.Textbox(label="子目录"),
            outputs=gr.Textbox(label="结果"),
            title="电影剧本文件名批量替换工具",
            description="无法初始化工具，请检查配置目录是否存在。"
        )
    
    def update_subdirectory_choices():
        """更新子目录选择框"""
        return gr.Dropdown(choices=renamer.get_subdirectories())
    
    def preview_rename_action(subdirectory):
        """预览重命名操作"""
        if not subdirectory:
            return "请选择一个子目录"
        results = renamer.preview_rename(subdirectory)
        return results
    
    def perform_rename_action(subdirectory):
        """执行重命名操作"""
        if not subdirectory:
            return "请选择一个子目录"
        results = renamer.perform_rename(subdirectory)
        return results
    
    with gr.Blocks(title="电影剧本文件名批量替换工具") as interface:
        gr.Markdown("# 电影剧本文件名批量替换工具")
        gr.Markdown("根据JSON映射文件将英文剧本文件名替换为中文")
        
        with gr.Row():
            with gr.Column(scale=1):
                subdirectory_dropdown = gr.Dropdown(
                    choices=renamer.get_subdirectories(),
                    label="选择子目录",
                    interactive=True
                )
                refresh_btn = gr.Button("刷新子目录列表", variant="secondary")
                
                with gr.Row():
                    preview_btn = gr.Button("预览重命名", variant="primary")
                    rename_btn = gr.Button("执行重命名", variant="danger")
            
            with gr.Column(scale=3):
                results_output = gr.JSON(label="操作结果")
        
        # 设置事件处理
        refresh_btn.click(
            fn=update_subdirectory_choices,
            inputs=[],
            outputs=[subdirectory_dropdown]
        )
        
        preview_btn.click(
            fn=preview_rename_action,
            inputs=[subdirectory_dropdown],
            outputs=[results_output]
        )
        
        rename_btn.click(
            fn=perform_rename_action,
            inputs=[subdirectory_dropdown],
            outputs=[results_output]
        )
    
    return interface

def apply_modifications(file_path: str, modifications: dict) -> bool:
    """
    根据 "行号: 内容" 的字典批量修改文件。
    为了效率和原子性，一次性读取文件，在内存中完成所有修改，然后一次性写回。

    Args:
        file_path (str): 目标文件的路径。
        modifications (dict): 一个字典，键是行号(int)，值是新的行内容(str)。

    Returns:
        bool: 如果所有修改都成功应用则返回 True，否则返回 False。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        mod_indices = {}
        for row_str, content in modifications.items():
            try:
                row = int(row_str)
                if row < 1 or row > len(lines):
                    print(f"错误: 行号 {row} 超出文件范围 (1-{len(lines)})。")
                    return False
                # 确保新内容以换行符结尾
                mod_indices[row - 1] = content if content.endswith('\\n') else content + '\\n'
            except ValueError:
                print(f"错误: 无效的行号 '{row_str}'。")
                return False

        for index, content in mod_indices.items():
            lines[index] = content

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        print(f"文件 {file_path} 已成功应用 {len(mod_indices)} 处修改。")
        return True
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return False
    except Exception as e:
        print(f"应用批量修改时发生错误: {e}")
        return False

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()    