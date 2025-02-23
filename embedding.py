from langchain.embeddings.base import Embeddings
from typing import List, Optional
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class LocalEmbedding(Embeddings):
    def __init__(self,
                 model_path: str,
                 device: Optional[str] = None,
                 quantize: bool = False,
                 max_batch_size: int = 32):
        """
        Args:
            model_path: 本地模型路径
            device: 指定设备 ("cuda", "cpu" 或 None 自动选择)
            quantize: 是否启用量化推理（FP16/BF16）
            max_batch_size: 最大批处理大小（防止OOM）
        """
        # 自动检测最佳设备
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 量化配置
        torch_dtype = None
        if quantize:
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16  # 优先使用BF16（更好的数值稳定性）
            else:
                torch_dtype = torch.float16  # 兼容旧显卡

        # 加载模型（启用自动设备映射）
        self.model = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            local_files_only=True,
        ).eval()  # 设置为评估模式

        # 加载Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

        # 性能参数
        self.max_batch_size = max_batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档向量"""
        # 自动分批处理（防止OOM）
        embeddings = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            embeddings += self._process_batch(batch)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """生成单个查询向量"""
        return self.embed_documents([text])[0]

    def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """处理单个批次"""
        # 编码文本
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.model.device)  # 输入数据与模型同设备

        # 推理（禁用梯度计算）
        with torch.no_grad():
            if torch.is_autocast_enabled() or self.model.dtype != torch.float32:
                with torch.autocast(device_type=self.device.split(':')[0], enabled=True):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        # 池化处理
        embeddings = self._mean_pooling(
            outputs.last_hidden_state,
            inputs["attention_mask"]
        )

        # 转换为CPU numpy数组（减少显存占用）
        return embeddings.cpu().float().numpy().tolist()

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """加权平均池化（保持FP32计算确保精度）"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings.float() * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

# class LocalEmbedding(Embeddings):
#     def __init__(self, model):
#         self.model = model  # 本地模型实例
#
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         # 批量处理文档
#         embeddings = self.model.encode(texts, convert_to_numpy=True)
#         return embeddings.tolist()
#
#     def embed_query(self, text: str) -> List[float]:
#         # 处理单个查询
#         return self.model.encode(text, convert_to_numpy=True).tolist()

# class LocalEmbedding(Embeddings):
#     def __init__(self, model, tokenizer, max_length=512):
#         self.model = model  # BGEM3FlagModel 实例
#         self.tokenizer = tokenizer  # XLMRobertaTokenizerFast 实例
#         self.max_length = max_length  # 最大输入长度
#
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         # 批量处理文档
#         inputs = self.tokenizer(
#             texts,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"  # 返回 PyTorch 张量
#         )
#         with torch.no_grad():
#             # 假设 BGEM3FlagModel 的调用方式是 model.encode(inputs)
#             embeddings = self.model.encode(**inputs)  # 根据实际模型调整
#         return embeddings.gpu().numpy().tolist()
#
#     def embed_query(self, text: str) -> List[float]:
#         # 处理单个查询
#         inputs = self.tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
#         with torch.no_grad():
#             # 假设 BGEM3FlagModel 的调用方式是 model.encode(inputs)
#             embedding = self.model.encode(**inputs)  # 根据实际模型调整
#         return embedding.gpu().numpy().tolist()[0]  # 返回单条嵌入