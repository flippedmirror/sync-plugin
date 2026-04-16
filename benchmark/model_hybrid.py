"""Hybrid inference: ONNX INT8 encoder + PyTorch decoder with KV caching.

The encoder (vision tower + text encoder) is pure feedforward — benefits from ONNX optimization.
The decoder is autoregressive — needs KV caching which PyTorch's generate() provides natively.
"""

from __future__ import annotations

import os
import time

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class FlorenceHybridModel:
    def __init__(
        self,
        onnx_dir: str,
        model_name: str = "microsoft/Florence-2-base-ft",
        use_int8: bool = True,
    ):
        suffix = "_int8" if use_int8 else ""
        enc_path = os.path.join(onnx_dir, f"encoder{suffix}.onnx")

        print(f"Loading hybrid model [encoder=ONNX int8={use_int8}, decoder=PyTorch]...")

        # ONNX encoder
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4
        sess_opts.inter_op_num_threads = 4
        self.encoder_session = ort.InferenceSession(enc_path, sess_options=sess_opts)

        # PyTorch decoder (load full model, but we only use the decoder part)
        self.pytorch_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True
        )
        self.pytorch_model.eval()
        self.decoder = self.pytorch_model.language_model
        self.device = "cpu"
        self.dtype = torch.float32

        # Processor
        proc_path = os.path.join(onnx_dir, "processor")
        if os.path.exists(proc_path):
            self.processor = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True)
        else:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        print("Hybrid model loaded.")

    def _run_inference(self, image: Image.Image, task_prompt: str, text_input: str | None = None) -> tuple[dict, float]:
        prompt = task_prompt if text_input is None else task_prompt + text_input

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        pixel_values_np = inputs["pixel_values"].numpy().astype(np.float32)
        input_ids_np = inputs["input_ids"].numpy().astype(np.int64)

        start = time.perf_counter()

        # Step 1: ONNX encoder (fast — no autoregressive, benefits from INT8)
        enc_hidden_np, enc_mask_np = self.encoder_session.run(
            None,
            {"pixel_values": pixel_values_np, "input_ids": input_ids_np},
        )

        # Step 2: Convert to PyTorch tensors for decoder
        enc_hidden = torch.from_numpy(enc_hidden_np)
        enc_mask = torch.from_numpy(enc_mask_np).long()

        # Step 3: PyTorch decoder with KV caching via generate()
        # Create a fake encoder output object
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=enc_hidden)

        with torch.inference_mode():
            generated_ids = self.decoder.generate(
                input_ids=None,
                encoder_outputs=encoder_outputs,
                attention_mask=enc_mask,
                max_new_tokens=256,
                num_beams=1,
                do_sample=False,
            )

        latency = time.perf_counter() - start

        generated_text = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )
        return parsed, latency

    @staticmethod
    def pixel_bbox_to_loc_tokens(bbox: list[float], image_size: tuple[int, int]) -> str:
        w, h = image_size
        locs = []
        for i, val in enumerate(bbox):
            dim = w if i % 2 == 0 else h
            loc = int(val * 1000 / dim)
            loc = max(0, min(999, loc))
            locs.append(f"<loc_{loc}>")
        return "".join(locs)

    def ocr(self, image: Image.Image) -> tuple[str, float]:
        result, latency = self._run_inference(image, "<OCR>")
        return result.get("<OCR>", ""), latency

    def ocr_with_region(self, image: Image.Image) -> tuple[dict, float]:
        result, latency = self._run_inference(image, "<OCR_WITH_REGION>")
        return result.get("<OCR_WITH_REGION>", {"quad_boxes": [], "labels": []}), latency

    def caption(self, image: Image.Image) -> tuple[str, float]:
        result, latency = self._run_inference(image, "<CAPTION>")
        return result.get("<CAPTION>", ""), latency

    def region_to_description(self, image: Image.Image, bbox_pixels: list[float]) -> tuple[str, float]:
        loc_str = self.pixel_bbox_to_loc_tokens(bbox_pixels, image.size)
        result, latency = self._run_inference(image, "<REGION_TO_DESCRIPTION>", text_input=loc_str)
        return result.get("<REGION_TO_DESCRIPTION>", ""), latency

    def open_vocabulary_detection(self, image: Image.Image, text: str) -> tuple[dict, float]:
        result, latency = self._run_inference(image, "<OPEN_VOCABULARY_DETECTION>", text_input=text)
        return result.get("<OPEN_VOCABULARY_DETECTION>", {"bboxes": [], "bboxes_labels": []}), latency

    def caption_to_phrase_grounding(self, image: Image.Image, caption: str) -> tuple[dict, float]:
        result, latency = self._run_inference(image, "<CAPTION_TO_PHRASE_GROUNDING>", text_input=caption)
        return result.get("<CAPTION_TO_PHRASE_GROUNDING>", {"bboxes": [], "labels": []}), latency
