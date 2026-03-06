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

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

DPI = 300
ZOOM = 1.1  # 拡大係数 (1.0 = 等倍)

ID_PHOTO_SIZES = {
    "標準（縦30mm × 横24mm (3.0cm×2.4cm)）":              (24.0, 30.0),
    "パスポート・マイナンバー（縦45mm × 横35mm (4.5cm×3.5cm)）": (35.0, 45.0),
    "履歴書用・大（縦55mm × 横40mm (5.5cm×4.0cm)）":        (40.0, 55.0),
}

# CUPS 用: 用紙名 → (横mm, 縦mm) の対応表（フォールバック用）
_CUPS_PAPER_SIZE_MM = {
    "A3":       (297.0, 420.0),
    "A4":       (210.0, 297.0),
    "A5":       (148.0, 210.0),
    "A6":       (105.0, 148.0),
    "B4":       (257.0, 364.0),
    "B5":       (182.0, 257.0),
    "Letter":   (215.9, 279.4),
    "Legal":    (215.9, 355.6),
    "Postcard": (100.0, 148.0),
    "L":        (89.0,  127.0),
    "2L":       (127.0, 178.0),
    "KG":       (102.0, 152.0),
}

# CUPS でサイズが取得できなかった場合のフォールバック
_FALLBACK_PAPER_SIZES = [
    ("L版（89mm × 127mm）",    (89.0,  127.0)),
    ("2L版（127mm × 178mm）",  (127.0, 178.0)),
    ("ハガキ（100mm × 148mm）", (100.0, 148.0)),
    ("A4（210mm × 297mm）",    (210.0, 297.0)),
]

_RESOLUTION_KIND_LABELS = {
    "Draft":  "下書き (Draft)",
    "Low":    "低品質 (Low)",
    "Medium": "標準 (Medium)",
    "High":   "高品質 (High)",
}


def _detect_wsl():
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


IS_WSL = _detect_wsl()

# pwsh (PowerShell 7+) を優先し、なければ powershell.exe にフォールバック
# pwsh はデフォルト UTF-8、powershell.exe は CP932（日本語 Windows）
_PS_EXE, _PS_ENCODING = ("pwsh", "utf-8") if shutil.which("pwsh") else ("powershell.exe", "cp932")


def mm_to_px(mm):
    return round(mm / 25.4 * DPI)


def _run_ps(script):
    """PowerShellスクリプトを実行し、stdout をデコードして返す。"""
    result = subprocess.run(
        [_PS_EXE, "-NoProfile", "-Command", script], capture_output=True,
    )
    return result.stdout.decode(_PS_ENCODING, errors="replace")


def _to_win_path(path):
    """WSL パスを Windows パスに変換する。"""
    return subprocess.run(
        ["wslpath", "-w", str(Path(path).resolve())],
        capture_output=True, text=True,
    ).stdout.strip()


def _choose(title, items, display=str, empty_msg="項目が見つかりませんでした。"):
    """リストからインタラクティブに 1 つ選ばせる。空なら None を返す。"""
    if not items:
        print(empty_msg)
        return None
    print(f"\n{title}")
    for i, item in enumerate(items, 1):
        print(f"  {i}. {display(item)}")
    while True:
        try:
            choice = int(input("番号を入力してください: "))
            if 1 <= choice <= len(items):
                return items[choice - 1]
        except ValueError:
            pass
        except EOFError:
            print("\n入力がありません。終了します。")
            sys.exit(1)
        print(f"  1〜{len(items)} の番号を入力してください。")


def select_from_menu(title, options):
    """辞書のキーを選ばせて対応する値を返す。"""
    chosen = _choose(title, list(options.items()), display=lambda x: x[0])
    return chosen[1] if chosen else None


def _input_mm(label):
    """正の数値（mm）を入力させる。"""
    while True:
        try:
            value = float(input(f"  {label} (mm): "))
            if value > 0:
                return value
        except ValueError:
            pass
        except EOFError:
            print("\n入力がありません。終了します。")
            sys.exit(1)
        print("  正の数を入力してください。")


def select_id_photo_size(paper_w_mm, paper_h_mm):
    """証明写真サイズを選ぶ。カスタムを選んだ場合は用紙サイズ内に収まる寸法を手入力させる。"""
    options = list(ID_PHOTO_SIZES.items()) + [("カスタム入力", None)]
    chosen = _choose("【証明写真のサイズを選んでください】", options, display=lambda x: x[0])
    if chosen is None:
        return None
    _, size = chosen
    if size is not None:
        return size
    print("\n【カスタムサイズを入力してください】")
    while True:
        w = _input_mm("横幅")
        h = _input_mm("縦幅")
        if w <= paper_w_mm and h <= paper_h_mm:
            return w, h
        print(f"  用紙サイズ（横{paper_w_mm}mm × 縦{paper_h_mm}mm）を超えています。再入力してください。")


def get_paper_sizes(printer_name):
    """プリンターがサポートする用紙サイズを (表示名, (w_mm, h_mm)) のリストで返す。"""
    if IS_WSL:
        safe = printer_name.replace("'", "''")
        output = _run_ps(f"""
Add-Type -AssemblyName System.Drawing
$pd = New-Object System.Drawing.Printing.PrintDocument
$pd.PrinterSettings.PrinterName = '{safe}'
$pd.PrinterSettings.PaperSizes | ForEach-Object {{ "$($_.PaperName),$($_.Width),$($_.Height)" }}
""")
        items = []
        for line in output.strip().splitlines():
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            name, w_str, h_str = parts
            try:
                w_mm = round(int(w_str) / 100 * 25.4, 1)
                h_mm = round(int(h_str) / 100 * 25.4, 1)
                if w_mm > 0 and h_mm > 0:
                    items.append((f"{name} ({w_mm}mm × {h_mm}mm)", (w_mm, h_mm)))
            except ValueError:
                continue
        return items or _FALLBACK_PAPER_SIZES
    else:
        result = subprocess.run(
            ["lpoptions", "-p", printer_name, "-l"], capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith("PageSize/"):
                _, values_str = line.split(":", 1)
                items = []
                for v in values_str.split():
                    name = v.lstrip("*")
                    dims = _CUPS_PAPER_SIZE_MM.get(name)
                    if dims:
                        items.append((f"{name} ({dims[0]}mm × {dims[1]}mm)", dims))
                if items:
                    return items
        return _FALLBACK_PAPER_SIZES


def select_paper_size(printer_name):
    chosen = _choose("【印刷用紙のサイズを選んでください】", get_paper_sizes(printer_name),
                     display=lambda x: x[0])
    return chosen[1] if chosen else _FALLBACK_PAPER_SIZES[0][1]


def get_printers():
    if IS_WSL:
        output = _run_ps("Get-Printer | Select-Object -ExpandProperty Name")
        return [line.strip() for line in output.strip().splitlines() if line.strip()]
    result = subprocess.run(["lpstat", "-a"], capture_output=True, text=True)
    return [line.split()[0] for line in result.stdout.splitlines() if line.strip()]


def get_trays(printer_name):
    if IS_WSL:
        safe = printer_name.replace("'", "''")
        output = _run_ps(f"""
Add-Type -AssemblyName System.Drawing
$pd = New-Object System.Drawing.Printing.PrintDocument
$pd.PrinterSettings.PrinterName = '{safe}'
$pd.PrinterSettings.PaperSources | ForEach-Object {{ $_.SourceName }}
""")
        return [line.strip() for line in output.strip().splitlines() if line.strip()]
    result = subprocess.run(["lpoptions", "-p", printer_name, "-l"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("InputSlot/"):
            _, values_str = line.split(":", 1)
            return [v.lstrip("*") for v in values_str.split()]
    return ["Auto"]


def get_qualities(printer_name):
    """(表示名, 識別子) のリストを返す。識別子は WSL2=インデックス文字列、CUPS=オプション文字列。"""
    if IS_WSL:
        safe = printer_name.replace("'", "''")
        output = _run_ps(f"""
Add-Type -AssemblyName System.Drawing
$pd = New-Object System.Drawing.Printing.PrintDocument
$pd.PrinterSettings.PrinterName = '{safe}'
$pd.PrinterSettings.PrinterResolutions | ForEach-Object {{ "$($_.Kind),$($_.X),$($_.Y)" }}
""")
        items = []
        for i, line in enumerate(output.strip().splitlines()):
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            kind, x, y = parts
            if kind == "Custom":
                display = f"{x}×{y} dpi"
            else:
                label = _RESOLUTION_KIND_LABELS.get(kind, kind)
                display = f"{label} ({x}×{y} dpi)" if int(x) > 0 else label
            items.append((display, str(i)))
        return items
    result = subprocess.run(["lpoptions", "-p", printer_name, "-l"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("print-quality/"):
            _, values_str = line.split(":", 1)
            labels = {"3": "下書き (Draft)", "4": "標準 (Normal)", "5": "高品質 (High)"}
            return [(labels.get(v.lstrip("*"), v.lstrip("*")), f"print-quality={v.lstrip('*')}")
                    for v in values_str.split()]
        if line.startswith("Resolution/"):
            _, values_str = line.split(":", 1)
            return [(v.lstrip("*"), f"Resolution={v.lstrip('*')}") for v in values_str.split()]
    return [("標準 (Normal)", "print-quality=4")]


def select_printer():
    return _choose("【プリンターを選んでください】", get_printers(),
                   empty_msg="プリンターが見つかりませんでした。")


def select_tray(printer_name):
    return _choose("【給紙トレイを選んでください】", get_trays(printer_name),
                   empty_msg="トレイ情報を取得できませんでした。")


def select_quality(printer_name):
    chosen = _choose("【印刷品質を選んでください】", get_qualities(printer_name),
                     display=lambda x: x[0], empty_msg="印刷品質情報を取得できませんでした。")
    return chosen[1] if chosen else None


def preview_image(image_path):
    if IS_WSL:
        safe = _to_win_path(image_path).replace("'", "''")
        subprocess.Popen([_PS_EXE, "-NoProfile", "-Command", f"Invoke-Item '{safe}'"])
    else:
        subprocess.Popen(["xdg-open", str(Path(image_path).resolve())])


def print_borderless(image_path, printer_name, paper_w_mm, paper_h_mm, tray, quality):
    if IS_WSL:
        _print_borderless_wsl(image_path, printer_name, paper_w_mm, paper_h_mm, tray, quality)
    else:
        _print_borderless_cups(image_path, printer_name, paper_w_mm, paper_h_mm, tray, quality)


def _print_borderless_wsl(image_path, printer_name, paper_w_mm, paper_h_mm, tray_name, quality_idx):
    w_hundredths = round(paper_w_mm / 25.4 * 100)
    h_hundredths = round(paper_h_mm / 25.4 * 100)
    safe_path    = _to_win_path(image_path).replace("'", "''")
    safe_printer = printer_name.replace("'", "''")
    tray_script  = (
        f"$src = $pd.PrinterSettings.PaperSources | Where-Object {{ $_.SourceName -eq '{tray_name.replace(chr(39), chr(39)*2)}' }} | Select-Object -First 1\n"
        f"if ($src) {{ $pd.DefaultPageSettings.PaperSource = $src }}"
        if tray_name else ""
    )

    ps_script = f"""
Add-Type -AssemblyName System.Drawing
$img = [System.Drawing.Image]::FromFile('{safe_path}')
$pd  = New-Object System.Drawing.Printing.PrintDocument
$pd.PrinterSettings.PrinterName = '{safe_printer}'
$pd.DefaultPageSettings.Margins = New-Object System.Drawing.Printing.Margins(0, 0, 0, 0)
$tw = {w_hundredths}
$th = {h_hundredths}
$match = $pd.PrinterSettings.PaperSizes | Where-Object {{ [Math]::Abs($_.Width - $tw) -le 10 -and [Math]::Abs($_.Height - $th) -le 10 }} | Select-Object -First 1
if ($match) {{ $pd.DefaultPageSettings.PaperSize = $match; Write-Host "用紙サイズ: $($match.PaperName)" }}
else {{ $pd.DefaultPageSettings.PaperSize = New-Object System.Drawing.Printing.PaperSize('Custom', $tw, $th); Write-Host "用紙サイズ: カスタム ($($tw/100*25.4)mm x $($th/100*25.4)mm)" }}
{tray_script}
$res = $pd.PrinterSettings.PrinterResolutions[{quality_idx}]
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
    result = subprocess.run(
        [_PS_EXE, "-NoProfile", "-Command", ps_script], capture_output=True,
    )
    if result.returncode == 0:
        print(result.stdout.decode(_PS_ENCODING, errors="replace").strip())
    else:
        print(f"印刷エラー: {result.stderr.decode(_PS_ENCODING, errors='replace').strip()}")


def _print_borderless_cups(image_path, printer_name, paper_w_mm, paper_h_mm, tray_name, quality_opt):
    cmd = ["lp", "-d", printer_name,
           "-o", f"media=Custom.{paper_w_mm}x{paper_h_mm}mm", "-o", "fit-to-page",
           "-o", quality_opt]
    if tray_name:
        cmd += ["-o", f"InputSlot={tray_name}"]
    cmd.append(str(Path(image_path).resolve()))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"印刷ジョブを送信しました。{result.stdout.strip()}")
    else:
        print(f"印刷エラー: {result.stderr.strip()}")


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
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half_w = (right - left) / 2 / ZOOM
    half_h = (bottom - top) / 2 / ZOOM
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
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("使い方: uv run main.py <入力画像> [<出力PNG>]")
        sys.exit(1)
    if not Path(sys.argv[1]).exists():
        print(f"エラー: {sys.argv[1]} が見つかりません")
        sys.exit(1)

    output_path = sys.argv[2] if len(sys.argv) == 3 else "/tmp/id_photo_output.png"

    printer = select_printer()
    if not printer:
        sys.exit(0)

    paper_w_mm, paper_h_mm = select_paper_size(printer)
    id_w_mm, id_h_mm       = select_id_photo_size(paper_w_mm, paper_h_mm)

    generate_id_photo(sys.argv[1], output_path, id_w_mm, id_h_mm, paper_w_mm, paper_h_mm)

    print("\nプレビューを開いています...")
    preview_image(output_path)

    tray    = select_tray(printer)
    quality = select_quality(printer)
    ok = _choose("【印刷してもよいですか？（用紙・電源を確認してください）】",
                 ["はい、印刷する", "いいえ、中止する"])
    if ok == "はい、印刷する":
        print_borderless(output_path, printer, paper_w_mm, paper_h_mm, tray, quality)
