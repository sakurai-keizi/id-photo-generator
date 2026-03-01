# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ultralytics>=8.0.0",
#   "Pillow>=10.0.0",
#   "numpy>=1.24.0",
# ]
# ///

# 使い方: uv run main.py <入力画像> <出力PNG>

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

DPI = 300
ID_W_MM, ID_H_MM = 24.0, 30.0   # 証明写真サイズ (mm)
L_W_MM,  L_H_MM  = 89.0, 127.0  # L版用紙サイズ (mm)
ZOOM = 1.1                        # 拡大係数 (1.0 = 等倍)


def mm_to_px(mm):
    return round(mm / 25.4 * DPI)


def generate_id_photo(input_path, output_path):
    img = Image.open(input_path).convert("RGB")
    W, H = img.size

    # YOLOv8 ポーズ推定で最初の人物を検出
    results = YOLO("yolov8n-pose.pt")(np.array(img), verbose=False)
    if not results or not len(results[0].keypoints.xy):
        raise ValueError("人物を検出できませんでした。")

    kpts     = results[0].keypoints.xy[0].cpu().numpy()
    conf     = results[0].keypoints.conf[0].cpu().numpy()
    bbox_top = results[0].boxes.xyxy[0][1].item()  # 頭頂部 y

    # 肩キーポイント (インデックス 5, 6) から下端・水平中心を算出
    shoulders = [kpts[i] for i in (5, 6) if conf[i] > 0.3]
    if not shoulders:
        raise ValueError("肩を検出できませんでした。")
    shoulder_y  = max(p[1] for p in shoulders)
    shoulder_cx = sum(p[0] for p in shoulders) / len(shoulders)
    span = shoulder_y - bbox_top  # 頭頂〜肩の距離

    # クロップ範囲: 頭上に 5%、肩下に 2% の余白、アスペクト比 2.4:3.0
    top    = max(0, bbox_top   - span * 0.05)
    bottom = min(H, shoulder_y + span * 0.02)
    crop_h = bottom - top
    crop_w = crop_h * (ID_W_MM / ID_H_MM)
    left   = max(0, shoulder_cx - crop_w / 2)
    right  = min(W, shoulder_cx + crop_w / 2)

    # ズーム: 中心固定でクロップ範囲を縮小 → 拡大表示
    cx, cy  = (left + right) / 2, (top + bottom) / 2
    half_w  = (right - left) / 2 / ZOOM
    half_h  = (bottom - top) / 2 / ZOOM
    left, right = max(0, cx - half_w), min(W, cx + half_w)
    top, bottom = max(0, cy - half_h), min(H, cy + half_h)

    # クロップ → リサイズ (アスペクト比保持, LANCZOS)
    cropped = img.crop((int(left), int(top), int(right), int(bottom)))
    cropped.thumbnail((mm_to_px(ID_W_MM), mm_to_px(ID_H_MM)), Image.LANCZOS)

    # L版白背景の左上に貼り付けて PNG 保存
    canvas = Image.new("RGB", (mm_to_px(L_W_MM), mm_to_px(L_H_MM)), "white")
    canvas.paste(cropped, (0, 0))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", dpi=(DPI, DPI))
    print(f"保存完了: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使い方: uv run main.py <入力画像> <出力PNG>")
        sys.exit(1)
    if not Path(sys.argv[1]).exists():
        print(f"エラー: {sys.argv[1]} が見つかりません")
        sys.exit(1)
    generate_id_photo(sys.argv[1], sys.argv[2])
