import os

def clean_special_chars_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 去除所有 form feed（\f）和其他你想去掉的特殊字符
    cleaned = content.replace('\f', '')
    # 如有其他特殊字符可继续 .replace('\u200b', '') 等
    if cleaned != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f"已清理特殊字符: {file_path}")

def clean_directory(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                file_path = os.path.join(dirpath, filename)
                clean_special_chars_in_file(file_path)

if __name__ == "__main__":
    clean_directory("/opt/rag_milvus_kb_project/kb_data/script/juben_cn")