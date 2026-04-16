"""CoreML inference wrapper for Florence-2 (encoder + decoder).

Uses CoreML's ANE/GPU for fast inference on Apple Silicon.
Implements greedy autoregressive decoding.
"""

from __future__ import annotations

import os
import time

import coremltools as ct
import numpy as np
from PIL import Image
from transformers import AutoProcessor


class FlorenceCoreMLModel:
    def __init__(self, model_dir: str, use_int8: bool = False):
        suffix = "int8" if use_int8 else "fp16"
        enc_path = os.path.join(model_dir, f"encoder_{suffix}.mlpackage")
        dec_path = os.path.join(model_dir, f"decoder_{suffix}.mlpackage")
        proc_path = os.path.join(model_dir, "processor")

        print(f"Loading CoreML models [{suffix}] from {model_dir}...")

        # compute_units: ALL = ANE + GPU + CPU (let CoreML decide)
        self.encoder = ct.models.MLModel(enc_path, compute_units=ct.ComputeUnit.ALL)
        self.decoder = ct.models.MLModel(dec_path, compute_units=ct.ComputeUnit.ALL)

        self.processor = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True)

        self.eos_token_id = self.processor.tokenizer.eos_token_id or 2
        self.decoder_start_id = self.processor.tokenizer.convert_tokens_to_ids("</s>")
        if self.decoder_start_id is None:
            self.decoder_start_id = 2

        print("CoreML models loaded.")

    def _encode(self, pixel_values: np.ndarray, input_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = self.encoder.predict({
            "pixel_values": pixel_values,
            "input_ids": input_ids.astype(np.int32),
        })
        return result["encoder_hidden_states"], result["encoder_attention_mask"]

    def _decode_step(self, dec_ids: np.ndarray, enc_hidden: np.ndarray, enc_mask: np.ndarray) -> np.ndarray:
        result = self.decoder.predict({
            "decoder_input_ids": dec_ids.astype(np.int32),
            "encoder_hidden_states": enc_hidden,
            "encoder_attention_mask": enc_mask,
        })
        return result["logits"]

    def _greedy_generate(self, enc_hidden: np.ndarray, enc_mask: np.ndarray, max_new_tokens: int = 256) -> list[int]:
        generated = [self.decoder_start_id]

        # Ensure consistent float types for decoder
        enc_hidden = enc_hidden.astype(np.float16)
        enc_mask = enc_mask.astype(np.float16)

        for step in range(max_new_tokens):
            dec_ids = np.array([generated], dtype=np.int32)
            try:
                logits = self._decode_step(dec_ids, enc_hidden, enc_mask)
            except RuntimeError:
                # CoreML can fail on certain sequence lengths; return what we have
                break
            next_token = int(np.argmax(logits[0, -1, :]))
            generated.append(next_token)
            if next_token == self.eos_token_id:
                break

        return generated

    def _run_inference(self, image: Image.Image, task_prompt: str, text_input: str | None = None) -> tuple[dict, float]:
        prompt = task_prompt if text_input is None else task_prompt + text_input

        inputs = self.processor(text=prompt, images=image, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32)
        input_ids = inputs["input_ids"].astype(np.int32)

        start = time.perf_counter()
        enc_hidden, enc_mask = self._encode(pixel_values, input_ids)
        generated_ids = self._greedy_generate(enc_hidden, enc_mask, max_new_tokens=256)
        latency = time.perf_counter() - start

        generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
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
