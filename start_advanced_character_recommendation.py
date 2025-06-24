#!/usr/bin/env python3
"""
高级人物推荐系统启动脚本
提供便捷的启动方式和配置选项
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_dependencies():
    """检查依赖项"""
    required_packages = [
        'pymilvus',
        'gradio', 
        'jieba',
        'python-dotenv',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_environment():
    """检查环境配置"""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ 未找到 .env 文件")
        print("请创建 .env 文件并配置以下环境变量:")
        print("MILVUS_URI=http://your-milvus-host:19530")
        print("PYTHONPATH=/path/to/your/project")
        print("AZURE_OPENAI_API_KEY=your_api_key")
        print("AZURE_OPENAI_ENDPOINT=your_endpoint")
        print("AZURE_EMBEDDING_DEPLOYMENT=your_deployment")
        return False
    
    print("✅ 环境配置文件存在")
    return True

def check_bm25_model():
    """检查BM25模型"""
    model_path = Path('src/utils/bm25_character_model.pkl')
    if not model_path.exists():
        print("⚠️  BM25模型文件不存在")
        print("建议运行以下命令训练BM25模型:")
        print("cd src/utils && python train_bm25_character_model.py")
        return False
    
    print("✅ BM25模型文件存在")
    return True

def start_system(port=7868, host="0.0.0.0", debug=False):
    """启动推荐系统"""
    print(f"🚀 启动高级人物推荐系统...")
    print(f"📍 访问地址: http://{host}:{port}")
    print(f"🔧 调试模式: {'开启' if debug else '关闭'}")
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = str(Path.cwd())
    
    # 启动应用
    app_path = Path('src/recommendation/character_recommendation_app.py')
    
    if not app_path.exists():
        print(f"❌ 应用文件不存在: {app_path}")
        return False
    
    try:
        # 使用subprocess启动，这样可以更好地处理信号
        cmd = [
            sys.executable, 
            str(app_path),
            '--port', str(port),
            '--host', host
        ]
        
        if debug:
            cmd.append('--debug')
        
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 系统已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    
    return True

def run_tests():
    """运行测试"""
    print("🧪 运行系统测试...")
    
    test_path = Path('src/recommendation/test_advanced_search.py')
    if not test_path.exists():
        print(f"❌ 测试文件不存在: {test_path}")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, str(test_path)
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='高级人物推荐系统启动器')
    parser.add_argument('--port', type=int, default=7868, help='服务端口 (默认: 7868)')
    parser.add_argument('--host', default='0.0.0.0', help='服务地址 (默认: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--test', action='store_true', help='运行测试')
    parser.add_argument('--check', action='store_true', help='检查环境配置')
    parser.add_argument('--train-bm25', action='store_true', help='训练BM25模型')
    
    args = parser.parse_args()
    
    print("🎭 高级人物推荐系统")
    print("=" * 50)
    
    # 检查环境
    if args.check:
        print("\n🔍 环境检查:")
        check_dependencies()
        check_environment()
        check_bm25_model()
        return
    
    # 训练BM25模型
    if args.train_bm25:
        print("\n🏋️ 训练BM25模型:")
        train_script = Path('src/utils/train_bm25_character_model.py')
        if train_script.exists():
            try:
                subprocess.run([sys.executable, str(train_script)])
            except Exception as e:
                print(f"❌ 训练失败: {e}")
        else:
            print(f"❌ 训练脚本不存在: {train_script}")
        return
    
    # 运行测试
    if args.test:
        print("\n🧪 运行测试:")
        run_tests()
        return
    
    # 启动前检查
    print("\n🔍 启动前检查:")
    if not check_dependencies():
        return
    
    if not check_environment():
        return
    
    check_bm25_model()  # 只是警告，不阻止启动
    
    # 启动系统
    print("\n🚀 启动系统:")
    start_system(args.port, args.host, args.debug)

if __name__ == "__main__":
    main() 