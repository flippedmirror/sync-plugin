import argparse

from benchmark.run import main

parser = argparse.ArgumentParser(
    description="Florence-2 cross-platform UI grounding benchmark"
)
parser.add_argument("--data-dir", required=True, help="Path to dataset directory with annotations.json")
parser.add_argument("--output-dir", default="results/", help="Output directory for results")
parser.add_argument(
    "--strategies", nargs="+",
    choices=["desc_only", "desc_ocr", "ocr_only"],
    default=None, help="Strategies to benchmark (default: all)",
)
parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Device (default: auto)")
parser.add_argument("--model", default="microsoft/Florence-2-base-ft", help="HuggingFace model ID")
parser.add_argument("--visualize", action="store_true", help="Save bbox visualizations")

args = parser.parse_args()
main(
    data_dir=args.data_dir,
    output_dir=args.output_dir,
    strategies=args.strategies,
    device=args.device,
    visualize=args.visualize,
    model_name=args.model,
)
