# API服务重启脚本使用说明

## 概述

`restart_apis.sh` 是一个用于管理推荐API服务的shell脚本，支持跨平台使用。

## 功能特性

- 🔄 自动重启所有API服务
- 🛑 智能停止占用端口的进程
- 📊 实时状态检查
- 📝 统一日志管理
- 🎨 彩色输出界面
- 🔧 支持多种操作模式

## 支持的服务

| 服务名称 | 脚本文件 | 端口 | 描述 |
|---------|---------|------|------|
| 综合推荐API | recommendation_api.py | 7003 | FastAPI综合推荐服务 |
| 人物推荐API | mcp_chara_api.py | 7012 | MCP人物推荐服务 |
| 场景推荐API | mcp_scene_api.py | 7013 | MCP场景推荐服务 |
| 剧本推荐API | mcp_script_api.py | 7014 | MCP剧本推荐服务 |

## 使用方法

### 1. 给脚本添加执行权限

```bash
chmod +x restart_apis.sh
```

### 2. 基本使用

```bash
# 重启所有服务（默认行为）
./restart_apis.sh

# 查看帮助信息
./restart_apis.sh -h

# 只检查服务状态
./restart_apis.sh -s

# 只停止所有服务
./restart_apis.sh -k
```

### 3. 操作示例

```bash
# 检查当前服务状态
./restart_apis.sh --status

# 停止所有服务
./restart_apis.sh --kill

# 重启所有服务
./restart_apis.sh
```

## 日志管理

- 所有服务的日志文件统一保存在 `logs/` 目录下
- 日志文件名格式：`{服务名}_{时间戳}.log`
- 示例：`mcp_chara_api_20241201_143022.log`

## 目录结构

```
rag_milvus_kb_project/
├── restart_apis.sh              # 重启脚本
├── logs/                        # 日志目录
│   ├── mcp_chara_api_*.log
│   ├── mcp_scene_api_*.log
│   ├── mcp_script_api_*.log
│   └── recommendation_api_*.log
└── src/recommendation/          # API脚本目录
    ├── mcp_chara_api.py
    ├── mcp_scene_api.py
    ├── mcp_script_api.py
    └── recommendation_api.py
```

## 注意事项

1. **权限要求**: 确保脚本有执行权限
2. **Python环境**: 确保Python环境已正确配置
3. **依赖包**: 确保所有必要的Python包已安装
4. **端口占用**: 脚本会自动处理端口冲突
5. **日志清理**: 定期清理旧的日志文件以节省磁盘空间

## 故障排除

### 常见问题

1. **权限不足**
   ```bash
   chmod +x restart_apis.sh
   ```

2. **Python路径问题**
   ```bash
   # 检查Python路径
   which python
   # 检查Python版本
   python --version
   ```

3. **端口被占用**
   ```bash
   # 手动检查端口占用
   lsof -i:7003
   lsof -i:7012
   lsof -i:7013
   lsof -i:7014
   ```

4. **服务启动失败**
   ```bash
   # 查看详细日志
   tail -f logs/*.log
   ```

## 开发说明

- 脚本使用相对路径，支持跨平台部署
- 自动检测项目根目录
- 支持GitHub等版本控制系统
- 彩色输出提升用户体验

## 更新日志

- v1.0: 初始版本，支持基本的重启功能
- v1.1: 添加状态检查和停止功能
- v1.2: 改进跨平台兼容性
- v1.3: 统一日志管理，优化用户体验 