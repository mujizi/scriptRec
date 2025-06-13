import os
from typing import List

def modify_line_in_file(file_path: str, line_number: int, new_content: str) -> bool:
    """
    修改文件中指定行的内容。

    Args:
        file_path (str): 目标文件的路径。
        line_number (int): 要修改的行号 (从1开始)。
        new_content (str): 新的行内容。

    Returns:
        bool: 如果成功则返回 True，否则返回 False。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if line_number < 1 or line_number > len(lines):
            print(f"错误: 行号 {line_number} 超出文件范围 (1-{len(lines)})。")
            return False

        lines[line_number - 1] = new_content + '\n'

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        return True
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return False
    except Exception as e:
        print(f"修改文件时发生错误: {e}")
        return False

def add_line_to_file(file_path: str, line_number: int, content: str) -> bool:
    """
    在文件中指定行号处添加新行。

    Args:
        file_path (str): 目标文件的路径。
        line_number (int): 要添加新行的位置 (从1开始)。
        content (str): 要添加的行内容。

    Returns:
        bool: 如果成功则返回 True，否则返回 False。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if line_number < 1 or line_number > len(lines) + 1:
            print(f"错误: 行号 {line_number} 超出可添加范围 (1-{len(lines) + 1})。")
            return False

        lines.insert(line_number - 1, content + '\n')

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        return True
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return False
    except Exception as e:
        print(f"添加行到文件时发生错误: {e}")
        return False

def delete_line_from_file(file_path: str, line_number: int) -> bool:
    """
    从文件中删除指定行。

    Args:
        file_path (str): 目标文件的路径。
        line_number (int): 要删除的行号 (从1开始)。

    Returns:
        bool: 如果成功则返回 True，否则返回 False。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if line_number < 1 or line_number > len(lines):
            print(f"错误: 行号 {line_number} 超出文件范围 (1-{len(lines)})。")
            return False

        del lines[line_number - 1]

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        return True
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return False
    except Exception as e:
        print(f"从文件删除行时发生错误: {e}")
        return False

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

        # 将所有待修改的行号转换为 0-based 索引
        # 并检查所有行号是否在有效范围内
        mod_indices = {}
        for row_str, content in modifications.items():
            try:
                row = int(row_str)
                if row < 1 or row > len(lines):
                    print(f"错误: 行号 {row} 超出文件范围 (1-{len(lines)})。")
                    return False
                mod_indices[row - 1] = content
            except ValueError:
                print(f"错误: 无效的行号 '{row_str}'。")
                return False

        # 在内存中应用修改
        for index, content in mod_indices.items():
            # 确保新内容以换行符结尾
            if not content.endswith('\n'):
                content += '\n'
            lines[index] = content

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        return True
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return False
    except Exception as e:
        print(f"应用批量修改时发生错误: {e}")
        return False
    