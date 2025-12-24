# --- START OF FILE local_vision_module.py ---
import logging
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

class LocalVisionModule:
    """本地视觉识别模块 - 使用 Qwen3-VL-8B"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        
        self.load_model()

    def load_model(self):
        """加载本地模型"""
        logger.info(f"正在加载本地视觉模型: {self.model_path} ...")
        try:
            # 加载模型
            # 如果显存不够，可以添加 load_in_4bit=True (需要安装 bitsandbytes)
            # 或者直接下载 GPTQ-Int4/AWQ 版本
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16, # 推荐 bf16，如果显卡不支持换 float16
                device_map="auto",          # 自动分配显卡/CPU
            )
            
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            logger.info("本地视觉模型加载完成！")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def identify_dish(self, image_path: str) -> str:
        """
        识别图片中的菜品
        """
        if not self.model:
            return "模型未加载"

        logger.info(f"正在分析图片: {image_path}")
        
        # 1. 准备输入数据
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": "你是一名专业的大厨，这是什么菜？请只回答菜名，不要包含其他废话。如果不确定，请描述主要食材。"},
                ],
            }
        ]

        # 2. 预处理
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # 3. 推理生成
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
        
        # 4. 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        dish_name = output_text[0].strip()
        logger.info(f"本地模型识别结果: {dish_name}")
        
        return dish_name