from __future__ import annotations

from model import form_answer

from tqdm import tqdm
import os
import argparse
import json


class Predictor:
    def __call__(self, audio_path: str):
        prediction = form_answer(audio_path)
        result = {
            "audio": os.path.basename(audio_path),          # Audio file base name
            "text": prediction.get("text", -1),             # Predicted text
            "label": prediction.get("label", -1),           # Text class
            "attribute": prediction.get("attribute", -1),   # Predicted attribute (if any, or -1)
        }
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get submission.")
    parser.add_argument(
        "--src",
        type=str,
        help="Path to the source audio files.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        help="Path to the output submission.",
    )
    args = parser.parse_args()
    predictor = Predictor()

    results = []
    for audio_path in tqdm(os.listdir(args.src)):

        result = predictor(os.path.join(args.src, audio_path))
        results.append(result)
    with open(
        os.path.join(args.dst, "submission.json"), "w", encoding="utf-8"
    ) as outfile:
        json.dump(results, outfile)

