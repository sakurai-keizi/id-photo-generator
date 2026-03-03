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
ZOOM = 1.1  # 拡大係数 (1.0 = 等倍)

ID_PHOTO_SIZES = {
    "標準（縦3.0cm × 横2.4cm）": (24.0, 30.0),
    "パスポート・マイナンバー（縦4.5cm × 横3.5cm）": (35.0, 45.0),
    "履歴書用・大（縦5.5cm × 横4.0cm）": (40.0, 55.0),
}

PAPER_SIZES = {
    "L版（89mm × 127mm）": (89.0, 127.0),
    "2L版（127mm × 178mm）": (127.0, 178.0),
    "ハガキ（100mm × 148mm）": (100.0, 148.0),
    "A4（210mm × 297mm）": (210.0, 297.0),
}


def mm_to_px(mm):
    return round(mm / 25.4 * DPI)


def select_from_menu(title, options):
    print(f"\n{title}")
    names = list(options.keys())
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    while True:
        try:
            choice = int(input("番号を入力してください: "))
            if 1 <= choice <= len(names):
                return options[names[choice - 1]]
        except (ValueError, EOFError):
            pass
        print(f"  1〜{len(names)} の番号を入力してください。")


def generate_id_photo(input_path, output_path, id_w_mm, id_h_mm, paper_w_mm, paper_h_mm):
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

    # クロップ範囲: 頭上に 5%、肩下に 2% の余白、指定アスペクト比
    top    = max(0, bbox_top   - span * 0.05)
    bottom = min(H, shoulder_y + span * 0.02)
    crop_h = bottom - top
    crop_w = crop_h * (id_w_mm / id_h_mm)
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
    cropped.thumbnail((mm_to_px(id_w_mm), mm_to_px(id_h_mm)), Image.LANCZOS)

    # 指定用紙サイズの白背景左上に貼り付けて PNG 保存
    canvas = Image.new("RGB", (mm_to_px(paper_w_mm), mm_to_px(paper_h_mm)), "white")
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

    id_w_mm, id_h_mm       = select_from_menu("【証明写真のサイズを選んでください】", ID_PHOTO_SIZES)
    paper_w_mm, paper_h_mm = select_from_menu("【印刷用紙のサイズを選んでください】", PAPER_SIZES)

    generate_id_photo(sys.argv[1], sys.argv[2], id_w_mm, id_h_mm, paper_w_mm, paper_h_mm)
