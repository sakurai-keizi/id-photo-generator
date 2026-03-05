# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ultralytics>=8.0.0",
#   "Pillow>=10.0.0",
#   "numpy>=1.24.0",
#   "torch>=2.0.0",
#   "torchvision>=0.15.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu" }]
# torchvision = [{ index = "pytorch-cpu" }]
# ///

# 使い方: uv run main.py <入力画像> <出力PNG>

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
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

# 各値は (Windows PrinterResolutionKind, CUPS print-quality)
QUALITY_OPTIONS = {
    "下書き（Draft）":  ("Draft",  "3"),
    "標準（Normal）":   ("Medium", "4"),
    "高品質（High）":   ("High",   "5"),
}


def mm_to_px(mm):
    return round(mm / 25.4 * DPI)



def is_wsl():
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def get_printers():
    if is_wsl():
        result = subprocess.run(
            ["powershell.exe", "-Command", "Get-Printer | Select-Object -ExpandProperty Name"],
            capture_output=True,
        )
        output = result.stdout.decode("cp932", errors="replace")
        return [line.strip() for line in output.strip().splitlines() if line.strip()]
    else:
        result = subprocess.run(["lpstat", "-a"], capture_output=True, text=True)
        return [line.split()[0] for line in result.stdout.splitlines() if line.strip()]


def get_trays(printer_name):
    if is_wsl():
        safe_printer = printer_name.replace("'", "''")
        ps_script = f"""
Add-Type -AssemblyName System.Drawing
$pd = New-Object System.Drawing.Printing.PrintDocument
$pd.PrinterSettings.PrinterName = '{safe_printer}'
$pd.PrinterSettings.PaperSources | ForEach-Object {{ $_.SourceName }}
"""
        result = subprocess.run(["powershell.exe", "-Command", ps_script], capture_output=True)
        output = result.stdout.decode("cp932", errors="replace")
        return [line.strip() for line in output.strip().splitlines() if line.strip()]
    else:
        result = subprocess.run(
            ["lpoptions", "-p", printer_name, "-l"], capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith("InputSlot/"):
                _, values_str = line.split(":", 1)
                return [v.lstrip("*") for v in values_str.split()]
        return ["Auto"]


def select_tray(printer_name):
    trays = get_trays(printer_name)
    if not trays:
        print("トレイ情報を取得できませんでした。")
        return None
    print("\n【給紙トレイを選んでください】")
    for i, name in enumerate(trays, 1):
        print(f"  {i}. {name}")
    while True:
        try:
            choice = int(input("番号を入力してください: "))
            if 1 <= choice <= len(trays):
                return trays[choice - 1]
        except ValueError:
            pass
        except EOFError:
            print("\n入力がありません。終了します。")
            sys.exit(1)
        print(f"  1〜{len(trays)} の番号を入力してください。")


def select_printer():
    printers = get_printers()
    if not printers:
        print("プリンターが見つかりませんでした。")
        return None
    print("\n【プリンターを選んでください】")
    for i, name in enumerate(printers, 1):
        print(f"  {i}. {name}")
    while True:
        try:
            choice = int(input("番号を入力してください: "))
            if 1 <= choice <= len(printers):
                return printers[choice - 1]
        except ValueError:
            pass
        except EOFError:
            print("\n入力がありません。終了します。")
            sys.exit(1)
        print(f"  1〜{len(printers)} の番号を入力してください。")


def preview_image(image_path):
    abs_path = str(Path(image_path).resolve())
    if is_wsl():
        win_path = subprocess.run(
            ["wslpath", "-w", abs_path], capture_output=True, text=True,
        ).stdout.strip()
        safe = win_path.replace("'", "''")
        subprocess.Popen(["powershell.exe", "-Command", f"Invoke-Item '{safe}'"])
    else:
        subprocess.Popen(["xdg-open", abs_path])


def print_borderless(image_path, printer_name, paper_w_mm, paper_h_mm, tray, quality):
    if is_wsl():
        _print_borderless_wsl(image_path, printer_name, paper_w_mm, paper_h_mm, tray, quality[0])
    else:
        _print_borderless_cups(image_path, printer_name, paper_w_mm, paper_h_mm, tray, quality[1])


def _print_borderless_wsl(image_path, printer_name, paper_w_mm, paper_h_mm, tray_name, quality_kind):
    win_path = subprocess.run(
        ["wslpath", "-w", str(Path(image_path).resolve())],
        capture_output=True, text=True,
    ).stdout.strip()

    # 用紙サイズ: mm → 1/100インチ単位（System.Drawing.Printing の単位）
    w_hundredths = round(paper_w_mm / 25.4 * 100)
    h_hundredths = round(paper_h_mm / 25.4 * 100)

    safe_path    = win_path.replace("'", "''")
    safe_printer = printer_name.replace("'", "''")
    safe_tray    = tray_name.replace("'", "''")

    ps_script = f"""
Add-Type -AssemblyName System.Drawing
$img = [System.Drawing.Image]::FromFile('{safe_path}')
$pd  = New-Object System.Drawing.Printing.PrintDocument
$pd.PrinterSettings.PrinterName   = '{safe_printer}'
$pd.DefaultPageSettings.Margins   = New-Object System.Drawing.Printing.Margins(0, 0, 0, 0)
$pd.DefaultPageSettings.PaperSize = New-Object System.Drawing.Printing.PaperSize('Custom', {w_hundredths}, {h_hundredths})
$src = $pd.PrinterSettings.PaperSources | Where-Object {{ $_.SourceName -eq '{safe_tray}' }} | Select-Object -First 1
if ($src) {{ $pd.DefaultPageSettings.PaperSource = $src }}
$res = $pd.PrinterSettings.PrinterResolutions | Where-Object {{ $_.Kind -eq [System.Drawing.Printing.PrinterResolutionKind]::{quality_kind} }} | Select-Object -First 1
if ($res) {{ $pd.DefaultPageSettings.PrinterResolution = $res }}
$imgRef = $img
$pd.add_PrintPage({{
    param($sender, $e)
    $e.Graphics.DrawImage($imgRef, $e.PageBounds)
}})
$pd.Print()
$img.Dispose()
Write-Host '印刷ジョブを送信しました。'
"""
    result = subprocess.run(["powershell.exe", "-Command", ps_script], capture_output=True)
    if result.returncode == 0:
        print(result.stdout.decode("cp932", errors="replace").strip())
    else:
        print(f"印刷エラー: {result.stderr.decode('cp932', errors='replace').strip()}")


def _print_borderless_cups(image_path, printer_name, paper_w_mm, paper_h_mm, tray_cups, quality_cups):
    media = f"Custom.{paper_w_mm}x{paper_h_mm}mm"
    result = subprocess.run(
        ["lp", "-d", printer_name,
         "-o", f"media={media}", "-o", "fit-to-page",
         "-o", f"InputSlot={tray_cups}", "-o", f"print-quality={quality_cups}",
         str(Path(image_path).resolve())],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"印刷ジョブを送信しました。{result.stdout.strip()}")
    else:
        print(f"印刷エラー: {result.stderr.strip()}")


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
        except ValueError:
            pass
        except EOFError:
            print("\n入力がありません。終了します。")
            sys.exit(1)
        print(f"  1〜{len(names)} の番号を入力してください。")


def generate_id_photo(input_path, output_path, id_w_mm, id_h_mm, paper_w_mm, paper_h_mm):
    Image.MAX_IMAGE_PIXELS = None  # 高解像度カメラ画像の制限を解除
    img = ImageOps.exif_transpose(Image.open(input_path)).convert("RGB")
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

    print("\nプレビューを開いています...")
    preview_image(sys.argv[2])

    printer = select_printer()
    if printer:
        tray    = select_tray(printer)
        quality = select_from_menu("【印刷品質を選んでください】", QUALITY_OPTIONS)
        print_borderless(sys.argv[2], printer, paper_w_mm, paper_h_mm, tray, quality)
