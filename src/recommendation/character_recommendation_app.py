import gradio as gr
from pymilvus import MilvusClient
from llm import embedding_func
import numpy as np
import uuid

# 连接到Milvus数据库（假设人物集合名为character_analysis）
db_name = "kb"
client = MilvusClient(uri="http://10.1.15.222:19530", db_name=db_name)
collection_name = "character2"  # 人物集合名
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

        print(f"Searching characters in '{collection_name}'...")
        search_res = client.search(
            collection_name=collection_name,
            data=emb,
            search_params=search_params,
            limit=top_k,
            anns_field="dense_vector",
            # 匹配人物数据结构字段
            output_fields=["id", "character_name", "basic_information", "characteristics", 
                          "biography", "character_summary", "script_name"],
        )

        results = []
        for hit in search_res[0]:
            similarity_score = hit["distance"]
            if similarity_score < SIMILARITY_THRESHOLD:
                continue
            
            entity = hit["entity"]
            results.append({
                "id": entity["id"],
                "character_name": entity["character_name"],
                "basic_information": entity["basic_information"],
                "characteristics": entity["characteristics"],
                "biography": entity["biography"],
                "character_summary": entity["character_summary"],
                "script_name": entity["script_name"],
                "similarity_score": similarity_score
            })
        print(f"Found {len(results)} matching characters.")
        return results

    except Exception as e:
        print(f"Search error: {e}")
        return []

def build_search_ui():
    def search(query):
        if not query.strip():
            return "请输入人物特征描述"
            
        results = vector_query(client, collection_name, query, top_k=5)
        
        if not results:
            return "未找到相关人物"
            
        output = "<div class='results-container'>"
        for i, result in enumerate(results, 1):
            similarity_percent = int(result['similarity_score'] * 100)
            color = f"rgb({255 - similarity_percent * 2.55}, {similarity_percent * 2.55}, 0)"
            
            output += f"""
            <div class="result-card" style="animation-delay: {i*0.1}s">
                <div class="card-header">
                    <h3 class="character-name">{result['character_name']}</h3>
                    <div class="similarity-badge" style="background-color: {color}">
                        相似度: {similarity_percent}%
                    </div>
                </div>
                <div class="card-content">
                    <div class="character-info">
                        <div class="info-row">
                            <span class="info-label">基本信息:</span>
                            <span class="info-value">{result['basic_information']}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">人物特征:</span>
                            <span class="info-value">{result['characteristics']}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">所属剧本:</span>
                            <span class="info-value">{result['script_name']}</span>
                        </div>
                    </div>
                    <div class="character-details">
                        <h4>人物传记</h4>
                        <p>{result['biography']}</p>
                        
                        <h4>人物总结</h4>
                        <p>{result['character_summary']}</p>
                    </div>
                </div>
            </div>
            """
        output += "</div>"
        return output

    with gr.Blocks(title="人物推荐系统", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <style>
            .results-container {display:flex;flex-direction:column;gap:20px;}
            .result-card {background:#fff;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.08);overflow:hidden;transform:translateY(10px);opacity:0;animation:fadeInUp 0.5s forwards;transition:all 0.3s ease;}
            .result-card:hover {box-shadow:0 8px 30px rgba(0,0,0,0.12);transform:translateY(-2px);}
            @keyframes fadeInUp {to{transform:translateY(0);opacity:1;}}
            .card-header {background:#f8f9fa;padding:16px 20px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #e9ecef;}
            .character-name {margin:0;color:#212529;font-size:1.2rem;font-weight:600;}
            .similarity-badge {padding:4px 12px;border-radius:12px;color:white;font-size:0.9rem;font-weight:500;}
            .card-content {padding:20px;}
            .character-info {display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:20px;}
            .info-row {display:flex;align-items:center;}
            .info-label {font-weight:500;color:#495057;min-width:80px;}
            .info-value {color:#212529;}
            .character-details h4 {color:#343a40;font-size:1rem;margin-top:15px;margin-bottom:5px;}
            .character-details p {color:#6c757d;margin:0 0 10px 0;line-height:1.5;}
            .gr-button {transition:all 0.3s ease;}
            .gr-button:hover {transform:translateY(-2px);}
            .gr-text-input:focus {box-shadow:0 0 0 3px rgba(13,110,253,0.25);}
        </style>
        """)
        
        gr.Markdown("""
        # <i class="fa-solid fa-user"></i> 人物推荐系统
        输入人物特征描述，系统将为你推荐最相关的虚构角色
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    query_input = gr.Textbox(
                        label="", 
                        placeholder="请输入人物特征（如：勇敢的太空探险家、复杂的反派角色）", 
                        lines=3,
                        container=False
                    )
                    search_btn = gr.Button("🔍 搜索", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("### <i class='fa-solid fa-info'></i> 使用说明")
                gr.Markdown("""
                - 支持自然语言描述（如：聪明机智的侦探）
                - 结果按语义相似度排序（阈值≥40%）
                - 展示人物背景、特征及所属剧本
                """)
                
        output = gr.HTML(label="搜索结果")
        
        with gr.Row():
            loading_status = gr.Markdown("", visible=False)
        
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
        
        gr.Examples(
            examples=[
                "勇敢坚韧的英雄角色",
                "性格勇敢，乐于助人的角色",
                "幽默风趣的科幻角色",
                "登山队队长",
                "年轻缉毒警察",
                "幽默的记者"
            ],
            inputs=query_input
        )

    return demo

if __name__ == "__main__":
    demo = build_search_ui()
    demo.launch(server_name="0.0.0.0", server_port=7864)