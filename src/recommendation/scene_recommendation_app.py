import gradio as gr
from pymilvus import MilvusClient
from src.utils.llm import embedding_func
import numpy as np
import uuid

# 连接到Milvus数据库
db_name = "kb"
client = MilvusClient(uri="http://124.221.215.17:19530", db_name=db_name)
collection_name = "scene_collection"  # 场景集合名
SIMILARITY_THRESHOLD = 0.4  # 相似度阈值

def vector_query(client, collection_name, text, top_k=5):
    try:
        if not client.has_collection(collection_name):
            print(f"Error: Collection '{collection_name}' does not exist.")
            return None
        emb = embedding_func([text])
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

    try:
        search_params = {
            "metric_type": "COSINE",
            "nprobe": 128,
        }

        print(f"Searching in collection '{collection_name}' for top {top_k} results...")
        search_res = client.search(
            collection_name=collection_name,
            data=emb,
            search_params=search_params,
            limit=top_k,
            anns_field="dense_vector",
            output_fields=["id", "scene_name", "scene_specifics", "scene_summary", "script_name"],
        )

        results = []
        for hit in search_res[0]:
            similarity_score = hit["distance"]
            
            # 检查相似度是否达到阈值
            if similarity_score < SIMILARITY_THRESHOLD:
                print(f"过滤掉相似度低的结果: {similarity_score}")
                continue
            
            entity = hit["entity"]
            results.append({
                "id": entity["id"],
                "scene_name": entity["scene_name"],
                "scene_specifics": entity["scene_specifics"],
                "scene_summary": entity["scene_summary"],
                "script_name": entity["script_name"],
                "similarity_score": similarity_score
            })
        print(f"Search successful. Found {len(results)} results.")
        return results

    except Exception as e:
        print(f"Error during search: {e}")
        return []

# 定义Gradio界面
def build_search_ui():
    def search(query):
        if not query.strip():
            return "请输入场景描述"
            
        results = vector_query(client, collection_name, query, top_k=5)
        
        if not results:
            return "未找到相关场景"
            
        # 构建美观的结果展示
        output = "<div class='results-container'>"
        for i, result in enumerate(results, 1):
            # 计算相似度百分比并转换为颜色
            similarity_percent = int(result['similarity_score'] * 100)
            # 根据相似度生成颜色，越相似越接近绿色
            color = f"rgb({255 - similarity_percent * 2.55}, {similarity_percent * 2.55}, 0)"
            
            output += f"""
            <div class="result-card" style="animation-delay: {i*0.1}s">
                <div class="card-header">
                    <h3 class="scene-name">{result['scene_name']}</h3>
                    <div class="similarity-badge" style="background-color: {color}">
                        相似度: {similarity_percent}%
                    </div>
                </div>
                <div class="card-content">
                    <div class="scene-info">
                        <div class="info-row">
                            <span class="info-label">所属剧本:</span>
                            <span class="info-value">{result['script_name']}</span>
                        </div>
                    </div>
                    <div class="scene-details">
                        <h4>场景详情</h4>
                        <p>{result['scene_specifics']}</p>
                        
                        <h4>场景总结</h4>
                        <p>{result['scene_summary']}</p>
                    </div>
                </div>
            </div>
            """
        output += "</div>"
        
        return output

    with gr.Blocks(title="场景推荐系统", theme=gr.themes.Soft()) as demo:
        # 自定义CSS
        gr.HTML("""
        <style>
            .results-container {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            
            .result-card {
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                overflow: hidden;
                transform: translateY(10px);
                opacity: 0;
                animation: fadeInUp 0.5s forwards;
                transition: all 0.3s ease;
            }
            
            .result-card:hover {
                box-shadow: 0 8px 30px rgba(0,0,0,0.12);
                transform: translateY(-2px);
            }
            
            @keyframes fadeInUp {
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }
            
            .card-header {
                background: #f8f9fa;
                padding: 16px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #e9ecef;
            }
            
            .scene-name {
                margin: 0;
                color: #212529;
                font-size: 1.2rem;
                font-weight: 600;
            }
            
            .similarity-badge {
                padding: 4px 12px;
                border-radius: 12px;
                color: white;
                font-size: 0.9rem;
                font-weight: 500;
            }
            
            .card-content {
                padding: 20px;
            }
            
            .scene-info {
                display: grid;
                grid-template-columns: 1fr;
                gap: 10px;
                margin-bottom: 20px;
            }
            
            .info-row {
                display: flex;
                align-items: center;
            }
            
            .info-label {
                font-weight: 500;
                color: #495057;
                min-width: 80px;
            }
            
            .info-value {
                color: #212529;
            }
            
            .scene-details h4 {
                color: #343a40;
                font-size: 1rem;
                margin-top: 15px;
                margin-bottom: 5px;
            }
            
            .scene-details p {
                color: #6c757d;
                margin: 0 0 10px 0;
                line-height: 1.5;
            }
            
            .gr-button {
                transition: all 0.3s ease;
            }
            
            .gr-button:hover {
                transform: translateY(-2px);
            }
            
            .gr-text-input {
                transition: all 0.3s ease;
            }
            
            .gr-text-input:focus {
                box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.25);
            }
        </style>
        """)
        
        gr.Markdown("""
        # <i class="fa-solid fa-street-view"></i> 场景推荐系统
        输入场景描述，系统将为你推荐最相关的影视场景
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    query_input = gr.Textbox(
                        label="", 
                        placeholder="请输入场景描述（如：雨夜的街头追逐、太空站中的危机）", 
                        lines=3,
                        container=False
                    )
                    search_btn = gr.Button("🔍 搜索", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("### <i class='fa-solid fa-lightbulb'></i> 使用提示")
                gr.Markdown("""
                - 输入场景类型、氛围、动作等描述
                - 系统将返回最相关的5个场景
                - 结果按相似度降序排列
                - 相似度以颜色直观展示
                """)
                
        output = gr.HTML(label="搜索结果")
        
        # 添加加载状态
        with gr.Row():
            loading_status = gr.Markdown("")
        
        search_btn.click(
            fn=lambda x: (gr.update(visible=True), None, gr.update(visible=False)),
            inputs=query_input,
            outputs=[loading_status, output, search_btn]
        ).then(
            fn=search,
            inputs=query_input,
            outputs=output
        ).then(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[loading_status, search_btn]
        )
        
        # 添加示例查询
        gr.Examples(
            examples=[
                "紧张刺激的汽车追逐",
                "浪漫的海滩日落场景",
                "充满悬疑的密室逃脱",
                "宏大的战争场面",
                "未来科技感的城市",
                "诡异的森林探险"
            ],
            inputs=query_input
        )

    return demo

if __name__ == "__main__":
    demo = build_search_ui()
    demo.launch(server_name="0.0.0.0", server_port=7865)