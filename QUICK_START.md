# 🚀 高级人物推荐系统 - 快速开始指南

## 📋 系统概述

这是一个基于Milvus向量数据库的高级人物推荐系统，集成了6种先进的检索策略：

- 🎯 **集成检索** - 综合多种策略，投票选出最佳结果
- 🔍 **稠密向量检索** - 基于语义相似度的深度检索  
- 📝 **BM25稀疏向量检索** - 基于关键词匹配的精确检索
- 🔄 **混合检索** - 结合稠密向量和BM25的优势
- 🧩 **语义分块检索** - 将查询分解为多个语义片段
- 📊 **多字段检索** - 在不同字段上分别搜索

## ⚡ 快速启动

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd rag_milvus_kb_project

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境配置

创建 `.env` 文件：

```bash
# Milvus配置
MILVUS_URI=http://10.1.15.222:19530
PYTHONPATH=/path/to/rag_milvus_kb_project

# Azure OpenAI配置
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_EMBEDDING_DEPLOYMENT=your_deployment
```

### 3. 一键启动

```bash
# 使用启动脚本
python start_advanced_character_recommendation.py

# 或者直接启动
cd src/recommendation
python character_recommendation_app.py
```

访问地址：http://localhost:7868

## 🎯 使用示例

### 基本搜索

1. 在搜索框中输入人物特征描述
2. 选择检索策略（推荐使用"集成检索"）
3. 调节返回结果数量
4. 点击"🔍 搜索"按钮

### 查询示例

```
✅ 推荐查询：
- 勇敢坚韧的英雄角色
- 幽默风趣的记者
- 复杂的反派角色
- 智慧型侦探
- 年轻缉毒警察
- 登山队队长
```

### 策略对比

点击"⚖️ 策略对比"按钮，系统会自动比较所有检索策略的效果，包括：
- 执行时间
- 结果数量
- 相似度分数
- 最佳匹配结果

## 🔧 高级功能

### BM25模型训练

```bash
# 训练BM25模型
python start_advanced_character_recommendation.py --train-bm25

# 或直接运行
cd src/utils
python train_bm25_character_model.py
```

### 系统测试

```bash
# 运行完整测试
python start_advanced_character_recommendation.py --test

# 或直接运行
cd src/recommendation
python test_advanced_search.py
```

### 环境检查

```bash
# 检查环境配置
python start_advanced_character_recommendation.py --check
```

## 📊 性能特点

### 检索策略对比

| 策略 | 优势 | 适用场景 | 平均耗时 |
|------|------|----------|----------|
| 集成检索 | 综合性强，准确率高 | 一般查询 | ~2.5s |
| 稠密向量 | 语义理解强 | 概念匹配 | ~1.2s |
| BM25稀疏向量 | 关键词精确 | 精确匹配 | ~0.8s |
| 混合检索 | 平衡性好 | 复杂查询 | ~2.0s |
| 语义分块 | 处理长查询 | 多主题查询 | ~3.0s |
| 多字段 | 字段精确 | 结构化查询 | ~1.5s |

### 系统特性

- ⚡ **并行处理**: 多线程并发执行检索策略
- 🧠 **智能去重**: 自动去除重复结果
- 📈 **性能监控**: 实时显示检索耗时
- 🎯 **置信度评分**: 为结果提供可信度指标
- 🔄 **策略对比**: 一键比较不同策略效果

## 🛠️ 故障排除

### 常见问题

1. **Milvus连接失败**
   ```bash
   # 检查Milvus服务状态
   curl http://10.1.15.222:19530/health
   ```

2. **BM25模型加载失败**
   ```bash
   # 重新训练模型
   python start_advanced_character_recommendation.py --train-bm25
   ```

3. **依赖包缺失**
   ```bash
   # 重新安装依赖
   pip install -r requirements.txt
   ```

### 日志分析

系统提供详细的日志输出：
- 检索策略执行情况
- 性能指标统计
- 错误信息追踪
- 调试信息输出

## 🎨 界面功能

### 主要组件

- **搜索输入框**: 支持自然语言描述
- **策略选择器**: 6种检索策略可选
- **参数调节器**: 返回结果数量控制
- **搜索按钮**: 执行检索操作
- **策略对比按钮**: 比较不同策略效果
- **结果展示区**: 格式化显示检索结果

### 结果展示

每个检索结果包含：
- 人物名称和相似度分数
- 基本信息和人物特征
- 所属剧本和人物传记
- 人物总结和置信度

## 🔮 扩展功能

### 自定义检索策略

```python
def custom_search_strategy(query, top_k):
    # 实现自定义检索逻辑
    pass

# 在ensemble_search中添加
strategies.append(custom_search_strategy)
```

### 参数调节

```python
# 混合检索权重调节
hybrid_search(query, top_k, dense_weight=0.7)  # 稠密向量权重70%

# 相似度阈值调节
SIMILARITY_THRESHOLD = 0.4  # 40%相似度阈值
```

## 📞 技术支持

- 📧 邮箱: support@example.com
- 💬 讨论区: GitHub Issues
- 📖 文档: README.md
- 🧪 测试: test_advanced_search.py

## 🎉 开始使用

现在您已经了解了系统的基本功能，可以开始使用高级人物推荐系统了！

```bash
# 启动系统
python start_advanced_character_recommendation.py

# 访问界面
# http://localhost:7868
```

祝您使用愉快！🎭 