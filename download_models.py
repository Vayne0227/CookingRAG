# download_models.py
import os
from modelscope import snapshot_download

# 创建模型存放目录
model_dir = "./models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# print("开始下载 Embedding 模型 (BGE-M3)...")
# # 下载 BGE-M3 (用于文本向量化)
# # 注意：ModelScope 上 BGE 的 ID 可能会变，这里使用 Xorbits 镜像或直接指定
# bge_path = snapshot_download('Xorbits/bge-m3', cache_dir=model_dir)
# print(f"BGE-M3 下载完成，路径: {bge_path}")

print("\n" + "="*50 + "\n")

print("开始下载 Qwen3-VL-8B-Instruct...")
# 下载 Qwen3-VL-8B-Instruct
qwen_path = snapshot_download('qwen/Qwen3-VL-8B-Instruct', cache_dir=model_dir)
print(f"Qwen3-VL-8B-Instruct 下载完成，路径: {qwen_path}")

print("\n所有模型下载完毕！")
print(f"请记下以上路径，并在主程序 main.py 中更新模型路径配置。")