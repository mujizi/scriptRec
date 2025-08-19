import os
import sys
from dotenv import load_dotenv

load_dotenv()

# 从环境变量获取并设置 PYTHONPATH
pythonpath = os.getenv("PYTHONPATH")
if pythonpath and pythonpath not in sys.path:
    sys.path.insert(0, pythonpath)

print(os.getenv("MILVUS_URI"))
print(os.getenv("PYTHONPATH"))




import gradio as gr
from pymilvus import MilvusClient
from src.utils.llm import embedding_func # Assuming this is your embedding function
import numpy as np
from openai import AzureOpenAI


# 加载环境变量


# 从环境变量获取配置
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
API_VERSION = os.getenv('API_VERSION', '2024-02-01')

if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT]):
    raise ValueError("Missing required environment variables: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT")

# ==============================================================================
# 1. Milvus & OpenAI Configuration
# ==============================================================================
# Milvus Client Setup
# IMPORTANT: Ensure the URI is correct and Milvus is running.
try:
    client = MilvusClient(uri="http://124.221.215.17:19530", db_name="kb")
    # Check server status
    client.get_load_state(collection_name="script_collection")
    print("Successfully connected to Milvus.")
except Exception as e:
    print(f"Failed to connect to Milvus. Please check the URI and server status. Error: {e}")
    # Exit if we can't connect to the database
    exit()

# Collection Name
COLLECTION_NAME = "script_collection"

# Similarity Threshold for Semantic Search
# This is a distance threshold for COSINE metric. A value of 0.6 means we only keep results
# with a cosine distance < 0.6, which corresponds to a cosine similarity > 0.4.
SIMILARITY_DISTANCE_THRESHOLD = 0.4

# ==============================================================================
# 2. Search Logic
# ==============================================================================

def query_to_keywords(query):
    """
    Uses GPT-4 to refine a user query into potent search keywords for BM25.
    """
    prompt = f"""
    You are an expert in extracting keywords for script searches. Analyze the user's query and pull out the most effective keywords.
    User Query: "{query}"

    Follow these instructions:
    1. Extract core themes (e.g., love, mystery, sci-fi, crime).
    2. Extract emotional tones (e.g., warm, tense, humorous, tragic), if any.
    3. Extract settings/elements (e.g., campus, workplace, ancient times, future), if any.
    4. Extract character relationships (e.g., teacher-student, lovers, friends, enemies), if any.
    5. Retain important adjectives and modifiers.
    6. Remove irrelevant stop words and filler phrases.

    Requirements:
    - Preserve the original intent but rephrase for optimal semantic search.
    - Prioritize words that indicate the script's genre, theme, and style.
    - Output should be a natural Chinese phrase, not just a pile of keywords.
    - Directly output the optimized search text without any explanation or formatting.

    Example:
    User Input: "A movie about cops and robbers"
    Your Output: "police bandit crime action"
    """
    try:
        openai_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=API_VERSION
        )
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5, # Lower temperature for more focused keyword extraction
            top_p=0.9,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Azure OpenAI for keyword extraction: {e}")
        return query # Fallback to the original query if API fails

def semantic_search(client, collection_name, text, top_k=5):
    """
    Performs a semantic search using dense vectors in Milvus.
    (Formerly 'dense_vector_query')
    """
    print("--- Starting Semantic Search ---")
    try:
        # Generate embedding for the query text
        query_embedding = embedding_func([text])
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

    try:
        # Define search parameters for dense vector search
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 128}}
        
        search_res = client.search(
            collection_name=collection_name,
            data=query_embedding,
            limit=top_k,
            search_params=search_params,
            anns_field="dense_vector",
            output_fields=["id", "script_name", "script_theme", "script_genre", "script_type", "script_subtypes", "script_background", "script_synopsis", "script_structure", "script_summary"],
        )

        results = []
        if search_res and search_res[0]:
            for hit in search_res[0]:
                # **CLARIFICATION**: Milvus returns 'distance' for COSINE metric.
                # similarity = 1 - distance. Lower distance means higher similarity.
                distance = hit['distance']
                
                # Filter out results that are not similar enough
                # if distance > SIMILARITY_DISTANCE_THRESHOLD:
                #     print(f"Filtering out low similarity result (distance: {distance:.4f})")
                #     continue
                
                similarity_score = 1.0 - distance
                
                entity = hit["entity"]
                results.append({
                    "id": entity["id"],
                    "script_name": entity["script_name"],
                    "script_theme": entity["script_theme"],
                    "script_genre": entity["script_genre"],
                    "script_type": entity["script_type"],
                    "script_subtypes": entity["script_subtypes"],
                    "script_background": entity["script_background"],
                    "script_synopsis": entity["script_synopsis"],
                    "script_structure": entity["script_structure"],
                    "script_summary": entity["script_summary"],
                    # Add the corrected similarity score (higher is better)
                    "similarity": similarity_score
                })
        print(f"Semantic search found {len(results)} results.")
        return results

    except Exception as e:
        print(f"Error during semantic search: {e}")
        return []

def keyword_search(client, collection_name, text, top_k=5):
    """
    Performs a keyword search using BM25 sparse vectors in Milvus.
    (Formerly 'bm25_query')
    """
    print("--- Starting Keyword Search ---")
    try:
        # Refine the query into keywords using the LLM
        keywords = query_to_keywords(text)
        print(f"Refined keywords for BM25: '{keywords}'")
        
        # BM25 search in Milvus requires the query to be a list of strings
        search_res = client.search(
            collection_name=collection_name,
            data=[keywords],
            limit=top_k,
            search_params={"metric_type": "BM25"},
            anns_field="sparse_vector", # Search against the sparse vector field
            output_fields=["id", "script_name", "script_theme", "script_genre", "script_type", "script_subtypes", "script_background", "script_synopsis", "script_structure", "script_summary"],
        )

        results = []
        if search_res and search_res[0]:
            for hit in search_res[0]:
                # For BM25, 'distance' is a relevance score where higher is better.
                relevance_score = hit['distance']
                entity = hit["entity"]
                results.append({
                    "id": entity["id"],
                    "script_name": entity["script_name"],
                    "script_theme": entity["script_theme"],
                    "script_genre": entity["script_genre"],
                    "script_type": entity["script_type"],
                    "script_subtypes": entity["script_subtypes"],
                    "script_background": entity["script_background"],
                    "script_synopsis": entity["script_synopsis"],
                    "script_structure": entity["script_structure"],
                    "script_summary": entity["script_summary"],
                    "relevance_score": relevance_score
                })
        print(f"Keyword search found {len(results)} results.")
        return results

    except Exception as e:
        print(f"Error during keyword search: {e}")
        return []

# ==============================================================================
# 3. Gradio UI & Formatting
# ==============================================================================

def format_results_html(results, search_type):
    """
    Generates beautiful HTML to display search results in Gradio.
    """
    if not results:
        return f"<div class='no-results'><i class='fas fa-ghost'></i><p>未能找到与查询匹配的剧本</p><span>请尝试更换关键词或描述</span></div>"
    
    # Sort results for consistent display (highest score first)
    if search_type == "智能匹配":
        results = sorted(results, key=lambda x: x.get('similarity', 0), reverse=True)
    elif search_type == "关键词匹配":
        results = sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)
            
    output_html = "<div class='results-container'>"
    for i, result in enumerate(results):
        score_label = ""
        score_badge_style = ""

        if search_type == "智能匹配":
            similarity_percent = int(result.get('similarity', 0) * 100)
            score_label = f"匹配度: {similarity_percent}%"
            # Dynamic color from red to green for similarity
            hue = similarity_percent * 1.2 # 0 -> 0 (red), 100 -> 120 (green)
            score_badge_style = f"background: hsl({hue}, 80%, 45%);"

        elif search_type == "关键词匹配":
            relevance = result.get('relevance_score', 0)
            score_label = f"相关性: {relevance:.2f}"
            # A stylish, fixed color for BM25 relevance
            score_badge_style = "background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);"
            
        # Safely get values from result dictionary
        script_name = result.get('script_name', 'N/A')
        script_theme = result.get('script_theme', 'N/A')
        script_genre = result.get('script_genre', 'N/A')
        script_type = result.get('script_type', 'N/A')
        script_subtypes = result.get('script_subtypes', 'N/A')
        script_background = result.get('script_background', 'N/A')
        script_synopsis = result.get('script_synopsis', 'N/A')
        script_structure = result.get('script_structure', 'N/A')
        script_summary = result.get('script_summary', 'N/A')

        output_html += f"""
        <div class="result-card" style="animation-delay: {i*0.1}s">
            <div class="card-header">
                <h3 class="script-name">{script_name}</h3>
                <div class="score-badge" style="{score_badge_style}">
                    {score_label}
                </div>
            </div>
            <div class="card-body">
                <div class="info-grid">
                    <div class="info-item"><i class="fas fa-bullseye"></i><strong>主题:</strong> <span>{script_theme}</span></div>
                    <div class="info-item"><i class="fas fa-theater-masks"></i><strong>类别:</strong> <span>{script_genre}</span></div>
                    <div class="info-item"><i class="fas fa-tags"></i><strong>类型:</strong> <span>{script_type}</span></div>
                    <div class="info-item"><i class="fas fa-tag"></i><strong>亚类型:</strong> <span>{script_subtypes}</span></div>
                </div>
                <div class="details-section">
                    <h4><i class="fas fa-scroll"></i> 剧本简介</h4>
                    <p>{script_synopsis}</p>
                    <h4><i class="fas fa-landmark"></i> 剧本背景</h4>
                    <p>{script_background}</p>
                    <h4><i class="fas fa-sitemap"></i> 剧本结构</h4>
                    <p>{script_structure}</p>
                    <h4><i class="fas fa-book-open"></i> 剧本摘要</h4>
                    <p>{script_summary}</p>
                </div>
            </div>
        </div>
        """
    output_html += "</div>"
    return output_html

def build_search_ui():
    """
    Constructs the Gradio web interface.
    """
    def search_controller(query):
        if not query or not query.strip():
            empty_message = "<div class='no-results'><i class='fas fa-keyboard'></i><p>请输入查询内容</p></div>"
            return empty_message, empty_message
        
        # Perform both searches
        semantic_results = semantic_search(client, COLLECTION_NAME, query, top_k=5)
        keyword_results = keyword_search(client, COLLECTION_NAME, query, top_k=5)
        
        # Format results into HTML
        semantic_html = format_results_html(semantic_results, "智能匹配")
        keyword_html = format_results_html(keyword_results, "关键词匹配")
            
        return semantic_html, keyword_html

    # Modern CSS for a premium look
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

    :root {
        --theme-bg: #F8F9FA;
        --card-bg: #FFFFFF;
        --text-primary: #212529;
        --text-secondary: #6C757D;
        --accent-color: #0d6efd;
        --border-color: #DEE2E6;
        --shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        --font-family: 'Noto Sans SC', sans-serif;
    }

    body, .gradio-container { background-color: var(--theme-bg) !important; font-family: var(--font-family); }
    h1, h2, h3, h4 { color: var(--text-primary); font-weight: 700; }
    p, span { color: var(--text-secondary); line-height: 1.6; }
    
    .main-title { text-align: center; color: var(--text-primary); font-size: 2.5rem; margin-bottom: 0.5rem; }
    .subtitle { text-align: center; color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 2rem; }

    .search-box { padding: 1.5rem; background: var(--card-bg); border-radius: 16px; box-shadow: var(--shadow); }
    .gr-button { transition: all 0.3s ease; }
    .gr-button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(13, 110, 253, 0.4); }

    .section-title {
        font-size: 1.75rem; color: var(--text-primary); margin-bottom: 20px;
        text-align: center; font-weight: 500; letter-spacing: 1px;
    }
    .section-title i { margin-right: 12px; color: var(--accent-color); }
    
    .results-container { display: flex; flex-direction: column; gap: 24px; }
    .result-card {
        background: var(--card-bg); border-radius: 16px; box-shadow: var(--shadow);
        overflow: hidden; opacity: 0; transform: translateY(20px);
        animation: fadeInUp 0.6s forwards cubic-bezier(0.25, 0.46, 0.45, 0.94);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .result-card:hover { transform: translateY(-5px) scale(1.01); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12); }

    @keyframes fadeInUp { to { opacity: 1; transform: translateY(0); } }

    .card-header {
        padding: 16px 24px; display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid var(--border-color); background-color: #FCFDFD;
    }
    .script-name { margin: 0; font-size: 1.3rem; font-weight: 700; color: #343A40; }
    .score-badge {
        padding: 6px 16px; border-radius: 16px; color: white;
        font-size: 0.9rem; font-weight: 500; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }

    .card-body { padding: 24px; }
    .info-grid {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 16px; margin-bottom: 24px;
    }
    .info-item { display: flex; align-items: center; font-size: 0.95rem; }
    .info-item i { margin-right: 8px; color: var(--accent-color); width: 16px; text-align: center; }
    .info-item strong { color: #495057; margin-right: 8px; }
    .info-item span { color: #212529; }
    
    .details-section h4 {
        color: #343A40; font-size: 1.1rem; margin-top: 20px; margin-bottom: 8px;
        padding-bottom: 5px; border-bottom: 2px solid var(--accent-color);
        display: inline-block;
    }
    .details-section h4 i { margin-right: 8px; }
    .details-section p { color: #495057; margin: 0 0 12px 0; }

    .no-results {
        text-align: center; color: var(--text-secondary); padding: 40px;
        background: var(--card-bg); border-radius: 16px; min-height: 200px;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
    }
    .no-results i { font-size: 3rem; margin-bottom: 1rem; color: #ced4da; }
    .no-results p { font-size: 1.2rem; color: var(--text-primary); margin:0; }
    .no-results span { font-size: 1rem; }
    """
    
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), css=css) as demo:
        gr.HTML("""
        <div class='main-title'><i class="fas fa-film"></i> 智能剧本发现引擎</div>
        <p class='subtitle'>输入您的灵感，探索无限的剧本世界。系统将通过智能匹配与关键词匹配双路引擎为您推荐。</p>
        """)

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Group(elem_classes="search-box"):
                    query_input = gr.Textbox(
                        label="",
                        placeholder="例如：一个关于时间旅行的爱情故事，充满了悬疑和反转...",
                        lines=3,
                        container=False
                    )
                    search_btn = gr.Button("🚀 开始探索", variant="primary", scale=1)
            
            with gr.Column(scale=2):
                 with gr.Group(elem_classes="search-box"):
                    gr.Markdown("#### <i class='fas fa-lightbulb'></i> 使用提示")
                    gr.Markdown(
                        """
                        - **智能匹配**: 理解您的深层意图，找到语义最相近的剧本。
                        - **关键词匹配**: 精准捕捉核心词汇，快速定位相关内容。
                        - 结果按匹配度/相关性从高到低排列。
                        """
                    )

        with gr.Row(variant='panel', equal_height=False):
            with gr.Column(min_width=400):
                gr.HTML("<h2 class='section-title'><i class='fas fa-brain'></i> 智能匹配 (Semantic Match)</h2>")
                semantic_output = gr.HTML(label="智能匹配结果")
            with gr.Column(min_width=400):
                gr.HTML("<h2 class='section-title'><i class='fas fa-key'></i> 关键词匹配 (Keyword Match)</h2>")
                keyword_output = gr.HTML(label="关键词匹配结果")
        
        # Add example queries
        gr.Examples(
            examples=[
                "一个孤独的宇航员在废弃空间站发现生命的故事",
                "现代都市里的办公室恋情，但夹杂着商业间谍的元素",
                "一部风格类似《银翼杀手》的赛博朋克侦探剧",
                "主角团意外获得超能力后，在校园里引发的一系列啼笑皆非的事件",
                "关于家庭亲情与救赎，氛围温暖治愈",
                "生化危机爆发后的末日求生，强调人性的考验"
            ],
            inputs=query_input,
            label="灵感激发 ✨"
        )
        
        search_btn.click(
            fn=search_controller,
            inputs=query_input,
            outputs=[semantic_output, keyword_output]
        )

    return demo

# ==============================================================================
# 4. Application Entry Point
# ==============================================================================
if __name__ == "__main__":
    app_ui = build_search_ui()
    app_ui.launch(server_name="0.0.0.0", inbrowser=True)
