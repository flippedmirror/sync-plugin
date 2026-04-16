"""Export Florence-2 encoder + decoder to CoreML (.mlpackage), with fp16 and INT8 variants.

Approach:
  1. Trace encoder (vision + text encoder) and decoder wrappers with torch.jit.trace
  2. Convert traced models to CoreML via coremltools
  3. Save fp16 versions
  4. Quantize weights to INT8 and save

Usage:
    python scripts/export_coreml.py --output-dir models/florence2-coreml
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

# Suppress coremltools torch version warning
import warnings
warnings.filterwarnings("ignore", message="Torch version")


def _load_florence():
    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base-ft", torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
    return model, processor


class EncoderWrapper(nn.Module):
    def __init__(self, florence_model):
        super().__init__()
        self.florence = florence_model

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor):
        inputs_embeds = self.florence.get_input_embeddings()(input_ids)
        image_features = self.florence._encode_image(pixel_values)
        inputs_embeds, attention_mask = self.florence._merge_input_ids_with_image_features(
            image_features, inputs_embeds
        )
        encoder_outputs = self.florence.language_model.model.encoder(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=False,
        )
        # encoder_outputs is a tuple; first element is last_hidden_state
        return encoder_outputs[0], attention_mask


class DecoderStepWrapper(nn.Module):
    """Single decoder forward pass (no KV cache for simplicity in CoreML)."""
    def __init__(self, florence_model):
        super().__init__()
        self.decoder = florence_model.language_model.model.decoder
        self.lm_head = florence_model.language_model.lm_head

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ):
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )
        logits = self.lm_head(decoder_outputs[0])
        return logits


def export_encoder_coreml(florence_model, processor, output_dir):
    import coremltools as ct

    print("Tracing encoder...")
    wrapper = EncoderWrapper(florence_model)
    wrapper.eval()

    from PIL import Image as PILImage
    dummy_img = PILImage.new("RGB", (768, 768), (128, 128, 128))
    # Use a longer prompt for tracing to cover the dynamic range
    inputs = processor(text="<OPEN_VOCABULARY_DETECTION>a blue sign with text", images=dummy_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]
    text_len = input_ids.shape[1]
    print(f"  Traced with input_ids shape: {input_ids.shape}")

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (pixel_values, input_ids))

    print("Converting encoder to CoreML...")
    from coremltools import RangeDim
    text_seq = RangeDim(lower_bound=1, upper_bound=512, default=text_len)

    enc_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="pixel_values", shape=pixel_values.shape),
            ct.TensorType(name="input_ids", shape=(1, text_seq)),
        ],
        outputs=[
            ct.TensorType(name="encoder_hidden_states"),
            ct.TensorType(name="encoder_attention_mask"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
    )

    fp16_path = os.path.join(output_dir, "encoder_fp16.mlpackage")
    enc_model.save(fp16_path)
    print(f"  Saved: {fp16_path}")

    # INT8 quantization
    print("  Quantizing encoder to INT8...")
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )
    op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    config = OptimizationConfig(global_config=op_config)
    enc_int8 = linear_quantize_weights(enc_model, config=config)

    int8_path = os.path.join(output_dir, "encoder_int8.mlpackage")
    enc_int8.save(int8_path)
    print(f"  Saved: {int8_path}")

    return fp16_path, int8_path


def export_decoder_coreml(florence_model, processor, output_dir):
    import coremltools as ct

    print("Tracing decoder...")
    wrapper = DecoderStepWrapper(florence_model)
    wrapper.eval()

    hidden_dim = florence_model.language_model.model.decoder.layers[0].self_attn.embed_dim
    enc_seq_len = 581  # 577 image tokens + 4 text tokens (typical for <OCR>)

    dummy_dec_ids = torch.tensor([[2, 0, 5]], dtype=torch.long)
    dummy_enc_hidden = torch.randn(1, enc_seq_len, hidden_dim)
    dummy_enc_mask = torch.ones(1, enc_seq_len)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_dec_ids, dummy_enc_hidden, dummy_enc_mask))

    print("Converting decoder to CoreML...")

    # Use flexible shapes for decoder_input_ids (variable length during generation)
    from coremltools import RangeDim
    dec_seq = RangeDim(lower_bound=1, upper_bound=512, default=3)
    enc_seq = RangeDim(lower_bound=1, upper_bound=1024, default=enc_seq_len)

    dec_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="decoder_input_ids", shape=(1, dec_seq)),
            ct.TensorType(name="encoder_hidden_states", shape=(1, enc_seq, hidden_dim)),
            ct.TensorType(name="encoder_attention_mask", shape=(1, enc_seq)),
        ],
        outputs=[
            ct.TensorType(name="logits"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
    )

    fp16_path = os.path.join(output_dir, "decoder_fp16.mlpackage")
    dec_model.save(fp16_path)
    print(f"  Saved: {fp16_path}")

    # INT8 quantization
    print("  Quantizing decoder to INT8...")
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )
    op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    config = OptimizationConfig(global_config=op_config)
    dec_int8 = linear_quantize_weights(dec_model, config=config)

    int8_path = os.path.join(output_dir, "decoder_int8.mlpackage")
    dec_int8.save(int8_path)
    print(f"  Saved: {int8_path}")

    return fp16_path, int8_path


def main(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Florence-2-base-ft...")
    florence_model, processor = _load_florence()

    export_encoder_coreml(florence_model, processor, output_dir)
    export_decoder_coreml(florence_model, processor, output_dir)

    # Save processor for runtime
    processor.save_pretrained(os.path.join(output_dir, "processor"))

    print(f"\nDone! CoreML models saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="models/florence2-coreml")
    args = parser.parse_args()
    main(args.output_dir)
