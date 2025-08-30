from langchain_huggingface import HuggingFaceEmbeddings

# 直接使用 BAAI 官方嵌入模型 ID（自动下载到本地缓存，无需手动管理路径）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",  # 英文嵌入模型（轻量且常用）
    # 若需中文模型，替换为 "BAAI/bge-base-zh-v1.5"
    model_kwargs={"device": "cpu"},  # 若有GPU，可改为 "cuda"
    encode_kwargs={"normalize_embeddings": True}  # 归一化向量（检索任务必需）
)

# 测试嵌入功能（验证模型是否正常加载）
text = "This is a test sentence for BGE embedding"
embedding = embeddings.embed_query(text)
print(f"模型加载成功！嵌入向量维度：{len(embedding)}")  # 正常输出应为 768（bge-base 模型维度）