import gradio as gr
import pandas as pd
import os
import sys
import asyncio

# 由于我们现在使用 `python -m` 从根目录运行，
# Python会自动处理路径，不再需要手动修改sys.path。
from src.analysis.screenplay_analyzer import analyze_screenplay
from src.utils.file_modifier import apply_modifications


# --- 全局变量和辅助函数 ---
# 使用相对路径来定位项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KB_DATA_TEST_PATH = os.path.join(project_root, "kb_data", "test")

def get_script_files():
    """获取测试目录下的所有txt文件"""
    if not os.path.exists(KB_DATA_TEST_PATH):
        os.makedirs(KB_DATA_TEST_PATH, exist_ok=True)
        print(f"警告: 测试目录 {KB_DATA_TEST_PATH} 不存在，已自动创建。请在该目录中放入剧本文件。")
        return []
    return [f for f in os.listdir(KB_DATA_TEST_PATH) if f.endswith('.txt')]

def read_file_lines(file_path):
    """读取文件并返回所有行"""
    if not file_path or not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

async def run_analysis(script_filename):
    """(异步) 运行剧本分析并准备Gradio输出"""
    if not script_filename:
        return (
            gr.update(value=None, visible=False), 
            gr.update(value="请先选择一个剧本文件。", visible=True),
            gr.update(visible=False), None, None
        )

    file_path = os.path.join(KB_DATA_TEST_PATH, script_filename)
    print(f"开始分析文件: {file_path}")

    modifications = await analyze_screenplay(file_path)

    if not modifications:
        return (
            gr.update(value=None, visible=False),
            gr.update(value="分析完成，未发现需要修改的场景名。", visible=True),
            gr.update(visible=False), file_path, modifications
        )

    lines = read_file_lines(file_path)
    results = []
    for line_num_str, new_content in modifications.items():
        line_num = int(line_num_str)
        original_content = lines[line_num - 1].strip() if line_num <= len(lines) else ""
        results.append([line_num, original_content, new_content.strip()])
    
    df = pd.DataFrame(results, columns=["行号", "原始内容", "建议修改内容"])

    return (
        gr.update(value=df, visible=True),
        gr.update(value=f"分析完成！发现 {len(modifications)} 处可优化的场景名。请在下方确认：", visible=True),
        gr.update(visible=True), file_path, modifications
    )

def apply_changes(file_path, modifications):
    """应用修改到文件并返回结果"""
    if not file_path or not modifications:
        return "没有待处理的修改。", "分析和修改流程已完成。"

    success = apply_modifications(file_path, modifications)

    if success:
        final_message = f"成功！修改已保存到原文件：\n{file_path}"
        updated_content = "".join(read_file_lines(file_path))
        return updated_content, final_message
    else:
        final_message = "错误：应用修改失败，请查看后台日志。"
        return "文件修改失败。", final_message

def view_script(script_filename):
    """在文本框中显示所选剧本的内容"""
    if not script_filename:
        return ""
    file_path = os.path.join(KB_DATA_TEST_PATH, script_filename)
    content = "".join(read_file_lines(file_path))
    return content

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="剧本场景名修正工具") as demo:
    state_file_path = gr.State(None)
    state_modifications = gr.State(None)

    gr.Markdown("# 剧本场景名智能修正工具")
    gr.Markdown("从下拉菜单中选择一个剧本文件，工具将自动分析不规范的场景名，并在您确认后进行修正。")

    with gr.Row():
        with gr.Column(scale=1):
            file_dropdown = gr.Dropdown(
                label="选择剧本文件",
                choices=get_script_files(),
                interactive=True
            )
            analyze_btn = gr.Button("开始分析", variant="primary")
            
            gr.Markdown("---")
            status_message = gr.Markdown("状态：请选择文件并开始分析。", visible=True)
            modification_df = gr.DataFrame(headers=["行号", "原始内容", "建议修改内容"], visible=False)
            apply_btn = gr.Button("确认并应用修改", variant="stop", visible=False)

        with gr.Column(scale=2):
            gr.Markdown("### 剧本内容预览")
            script_view = gr.Textbox(
                label="剧本内容", 
                lines=30, 
                max_lines=30, 
                interactive=False, 
                show_copy_button=True
            )
    
    final_status = gr.Markdown("")

    # --- 事件处理 ---
    file_dropdown.change(fn=view_script, inputs=[file_dropdown], outputs=[script_view])
    analyze_btn.click(
        fn=run_analysis,
        inputs=[file_dropdown],
        outputs=[modification_df, status_message, apply_btn, state_file_path, state_modifications]
    )
    apply_btn.click(
        fn=apply_changes,
        inputs=[state_file_path, state_modifications],
        outputs=[script_view, final_status]
    ).then(lambda: (gr.update(visible=False), gr.update(visible=False)), None, [modification_df, apply_btn])

if __name__ == "__main__":
    # 允许从外部访问
    demo.launch(server_name="0.0.0.0") 