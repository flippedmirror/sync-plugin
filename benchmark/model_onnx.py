"""ONNX Runtime inference wrapper for Florence-2.

Implements greedy autoregressive decoding using the exported encoder + decoder ONNX models.
"""

from __future__ import annotations

import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor


class FlorenceONNXModel:
    def __init__(self, model_dir: str, use_int8: bool = True):
        suffix = "_int8" if use_int8 else ""
        enc_path = os.path.join(model_dir, f"encoder{suffix}.onnx")
        dec_path = os.path.join(model_dir, f"decoder{suffix}.onnx")
        proc_path = os.path.join(model_dir, "processor")

        print(f"Loading ONNX models from {model_dir} [int8={use_int8}]...")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.inter_op_num_threads = 4
        sess_opts.intra_op_num_threads = 4

        self.encoder_session = ort.InferenceSession(enc_path, sess_options=sess_opts)
        self.decoder_session = ort.InferenceSession(dec_path, sess_options=sess_opts)
        self.processor = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True)

        # Get special token IDs
        self.bos_token_id = self.processor.tokenizer.bos_token_id or 0
        self.eos_token_id = self.processor.tokenizer.eos_token_id or 2
        self.pad_token_id = self.processor.tokenizer.pad_token_id or 1

        print("ONNX models loaded.")

    def _encode(self, pixel_values: np.ndarray, input_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run encoder: pixel_values + input_ids → encoder_hidden_states + attention_mask."""
        outputs = self.encoder_session.run(
            None,
            {"pixel_values": pixel_values, "input_ids": input_ids},
        )
        return outputs[0], outputs[1]  # encoder_hidden_states, encoder_attention_mask

    def _decode_step(
        self, decoder_input_ids: np.ndarray, enc_hidden: np.ndarray, enc_mask: np.ndarray
    ) -> np.ndarray:
        """Run decoder: decoder_input_ids + encoder outputs → logits."""
        outputs = self.decoder_session.run(
            None,
            {
                "decoder_input_ids": decoder_input_ids,
                "encoder_hidden_states": enc_hidden,
                "encoder_attention_mask": enc_mask,
            },
        )
        return outputs[0]  # logits [batch, seq_len, vocab]

    def _greedy_generate(
        self, enc_hidden: np.ndarray, enc_mask: np.ndarray, max_new_tokens: int = 256
    ) -> list[int]:
        """Greedy autoregressive decoding."""
        # Start with decoder_start_token_id (usually </s> = 2 for BART)
        decoder_start = self.processor.tokenizer.convert_tokens_to_ids("</s>")
        if decoder_start is None:
            decoder_start = self.bos_token_id

        generated = [decoder_start]

        for _ in range(max_new_tokens):
            dec_ids = np.array([generated], dtype=np.int64)
            logits = self._decode_step(dec_ids, enc_hidden, enc_mask)
            next_token = int(np.argmax(logits[0, -1, :]))
            generated.append(next_token)
            if next_token == self.eos_token_id:
                break

        return generated

    def _run_inference(self, image: Image.Image, task_prompt: str, text_input: str | None = None) -> tuple[dict, float]:
        """Full inference pipeline matching FlorenceModel interface."""
        prompt = task_prompt if text_input is None else task_prompt + text_input

        inputs = self.processor(text=prompt, images=image, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32)
        input_ids = inputs["input_ids"].astype(np.int64)

        start = time.perf_counter()

        # Encode
        enc_hidden, enc_mask = self._encode(pixel_values, input_ids)

        # Decode (greedy)
        generated_ids = self._greedy_generate(enc_hidden, enc_mask, max_new_tokens=256)

        latency = time.perf_counter() - start

        # Post-process
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

    # Public API — same interface as FlorenceModel

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
