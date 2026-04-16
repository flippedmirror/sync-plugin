"""Export Florence-2-base-ft to ONNX (encoder + decoder) and quantize to INT8.

We export 2 ONNX models:
  - encoder.onnx: pixel_values + input_ids → encoder_hidden_states + attention_mask
  - decoder.onnx: decoder_input_ids + encoder_hidden_states + encoder_attention_mask → logits

Then quantize both to INT8.

Usage:
    python scripts/export_onnx.py --output-dir models/florence2-onnx
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor


class EncoderWrapper(nn.Module):
    """Wraps Florence-2's full encoder path: image → features + text → embeds → merge → encoder."""

    def __init__(self, florence_model):
        super().__init__()
        self.florence = florence_model

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor):
        # Exactly replicates the generate() flow:
        # 1. Text embeddings via the shared embedding layer
        inputs_embeds = self.florence.get_input_embeddings()(input_ids)

        # 2. Image features via vision tower + projection
        image_features = self.florence._encode_image(pixel_values)

        # 3. Merge: [image_features, text_embeds]
        inputs_embeds, attention_mask = self.florence._merge_input_ids_with_image_features(
            image_features, inputs_embeds
        )

        # 4. Run through the language model's text encoder
        encoder_outputs = self.florence.language_model.model.encoder(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return encoder_outputs.last_hidden_state, attention_mask


class DecoderWrapper(nn.Module):
    """Wraps Florence-2's decoder for single-step logit prediction (no KV cache)."""

    def __init__(self, florence_model):
        super().__init__()
        self.language_model = florence_model.language_model

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ):
        decoder_outputs = self.language_model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        logits = self.language_model.lm_head(decoder_outputs.last_hidden_state)
        return logits


def export_encoder(florence_model, processor, output_dir: str) -> str:
    print("Exporting encoder...")
    wrapper = EncoderWrapper(florence_model)
    wrapper.eval()

    from PIL import Image
    dummy_image = Image.new("RGB", (768, 768), (128, 128, 128))
    inputs = processor(text="<OCR>", images=dummy_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]

    path = os.path.join(output_dir, "encoder.onnx")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (pixel_values, input_ids),
            path,
            input_names=["pixel_values", "input_ids"],
            output_names=["encoder_hidden_states", "encoder_attention_mask"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "input_ids": {0: "batch", 1: "text_len"},
                "encoder_hidden_states": {0: "batch", 1: "enc_len"},
                "encoder_attention_mask": {0: "batch", 1: "enc_len"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved: {path} ({size_mb:.1f} MB)")
    return path


def export_decoder(florence_model, processor, output_dir: str) -> str:
    print("Exporting decoder...")
    wrapper = DecoderWrapper(florence_model)
    wrapper.eval()

    hidden_dim = florence_model.language_model.model.decoder.layers[0].self_attn.embed_dim
    print(f"  hidden_dim={hidden_dim}")

    # Dummy inputs
    dummy_dec_ids = torch.tensor([[2, 0, 5]], dtype=torch.long)  # short sequence
    dummy_enc_hidden = torch.randn(1, 600, hidden_dim)
    dummy_enc_mask = torch.ones(1, 600)

    path = os.path.join(output_dir, "decoder.onnx")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_dec_ids, dummy_enc_hidden, dummy_enc_mask),
            path,
            input_names=["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "decoder_input_ids": {0: "batch", 1: "dec_len"},
                "encoder_hidden_states": {0: "batch", 1: "enc_len"},
                "encoder_attention_mask": {0: "batch", 1: "enc_len"},
                "logits": {0: "batch", 1: "dec_len"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved: {path} ({size_mb:.1f} MB)")
    return path


def quantize_model(onnx_path: str, output_path: str):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    print(f"  Quantizing {os.path.basename(onnx_path)}...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )
    orig = os.path.getsize(onnx_path) / 1024 / 1024
    quant = os.path.getsize(output_path) / 1024 / 1024
    print(f"  {orig:.1f} MB → {quant:.1f} MB ({quant/orig:.0%})")


def main(output_dir: str, model_name: str = "microsoft/Florence-2-base-ft"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    enc_path = export_encoder(model, processor, output_dir)
    dec_path = export_decoder(model, processor, output_dir)

    print("\nQuantizing to INT8...")
    quantize_model(enc_path, os.path.join(output_dir, "encoder_int8.onnx"))
    quantize_model(dec_path, os.path.join(output_dir, "decoder_int8.onnx"))

    # Save processor for runtime use
    processor.save_pretrained(os.path.join(output_dir, "processor"))

    print(f"\nDone! Models in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="models/florence2-onnx")
    parser.add_argument("--model", default="microsoft/Florence-2-base-ft")
    args = parser.parse_args()
    main(args.output_dir, args.model)
