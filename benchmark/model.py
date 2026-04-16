from __future__ import annotations

import time

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

DEFAULT_MODEL = "microsoft/Florence-2-base-ft"


class FlorenceModel:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, fast: bool = False, compile_model: bool = False, image_size: int | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.dtype = torch.float16 if device in ("cuda:0", "mps") else torch.float32
        self.fast = fast

        print(f"Loading {model_name} on {device} ({self.dtype}) [fast={fast}, compile={compile_model}, image_size={image_size or 768}]...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        if compile_model:
            print("Compiling model with torch.compile (max-autotune)...")
            self.model = torch.compile(self.model, mode="max-autotune")

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Patch image processor resolution if requested
        if image_size is not None and image_size != 768:
            self.processor.image_processor.size = {"height": image_size, "width": image_size}
            self.processor.image_processor.crop_size = {"height": image_size, "width": image_size}
            print(f"  Patched image processor to {image_size}x{image_size}")

        print("Model loaded.")

    def _run_inference(self, image: Image.Image, task_prompt: str, text_input: str | None = None) -> tuple[dict, float]:
        """Core inference: process → generate → post-process. Returns (parsed_result, latency_seconds)."""
        prompt = task_prompt if text_input is None else task_prompt + text_input

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device, self.dtype)

        # fast mode: greedy decoding + shorter output for lower latency
        gen_kwargs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
        }
        if self.fast:
            gen_kwargs["max_new_tokens"] = 256
            gen_kwargs["num_beams"] = 1
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["max_new_tokens"] = 1024
            gen_kwargs["num_beams"] = 3

        start = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(**gen_kwargs)
        latency = time.perf_counter() - start

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )
        return parsed, latency

    @staticmethod
    def pixel_bbox_to_loc_tokens(bbox: list[float], image_size: tuple[int, int]) -> str:
        """Convert [x1, y1, x2, y2] pixel coords to Florence-2 loc token string."""
        w, h = image_size
        locs = []
        for i, val in enumerate(bbox):
            dim = w if i % 2 == 0 else h
            loc = int(val * 1000 / dim)
            loc = max(0, min(999, loc))
            locs.append(f"<loc_{loc}>")
        return "".join(locs)

    def region_to_description(self, image: Image.Image, bbox_pixels: list[float]) -> tuple[str, float]:
        """Describe the content at a bounding box. Returns (description, latency)."""
        loc_str = self.pixel_bbox_to_loc_tokens(bbox_pixels, image.size)
        result, latency = self._run_inference(image, "<REGION_TO_DESCRIPTION>", text_input=loc_str)
        desc = result.get("<REGION_TO_DESCRIPTION>", "")
        return desc, latency

    def ocr(self, image: Image.Image) -> tuple[str, float]:
        """Run plain OCR on image. Returns (text_string, latency)."""
        result, latency = self._run_inference(image, "<OCR>")
        return result.get("<OCR>", ""), latency

    def ocr_with_region(self, image: Image.Image) -> tuple[dict, float]:
        """Run OCR on image. Returns ({'quad_boxes': [...], 'labels': [...]}, latency)."""
        result, latency = self._run_inference(image, "<OCR_WITH_REGION>")
        return result.get("<OCR_WITH_REGION>", {"quad_boxes": [], "labels": []}), latency

    def caption(self, image: Image.Image) -> tuple[str, float]:
        """Generate a short caption for the image. Returns (caption_text, latency)."""
        result, latency = self._run_inference(image, "<CAPTION>")
        return result.get("<CAPTION>", ""), latency

    def open_vocabulary_detection(self, image: Image.Image, text: str) -> tuple[dict, float]:
        """Find elements matching text description. Returns ({'bboxes': [...], 'bboxes_labels': [...]}, latency)."""
        result, latency = self._run_inference(image, "<OPEN_VOCABULARY_DETECTION>", text_input=text)
        return result.get("<OPEN_VOCABULARY_DETECTION>", {"bboxes": [], "bboxes_labels": []}), latency

    def caption_to_phrase_grounding(self, image: Image.Image, caption: str) -> tuple[dict, float]:
        """Ground noun phrases in caption to bboxes. Returns ({'bboxes': [...], 'labels': [...]}, latency)."""
        result, latency = self._run_inference(image, "<CAPTION_TO_PHRASE_GROUNDING>", text_input=caption)
        return result.get("<CAPTION_TO_PHRASE_GROUNDING>", {"bboxes": [], "labels": []}), latency
