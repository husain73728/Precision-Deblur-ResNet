import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.utils as vutils

from model import DeblurResNet
from losses import psnr_batch, ssim_batch

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_image(path):
    return TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0)


def save_image(tensor, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(tensor.squeeze(0).clamp(0, 1)).save(path)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = DeblurResNet(n_features=cfg.get("n_features", 96), n_blocks=cfg.get("n_blocks", 8)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch','?')} best_psnr={ckpt.get('best_psnr', float('nan')):.2f}")
    return model


@torch.no_grad()
def deblur(model, img_tensor, device):
    return model(img_tensor.to(device)).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pth")
    p.add_argument("--input", default="blurred_images")
    p.add_argument("--sharp", default="sharp_images")
    p.add_argument("--out", default="results")
    p.add_argument("--save_grid", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)

    if input_path.is_file():
        blurred = load_image(input_path)
        restored = deblur(model, blurred, device)
        save_image(restored, out_dir / f"restored_{input_path.name}")
        print(f"Saved {out_dir / f'restored_{input_path.name}'}")
        return

    files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in EXTS)
    sharp_dir = Path(args.sharp) if args.sharp else None
    psnrs, ssims = [], []

    for f in files:
        blurred = load_image(f)
        restored = deblur(model, blurred, device)
        save_image(restored, out_dir / f"restored_{f.name}")

        if sharp_dir and (sharp_dir / f.name).exists():
            sharp = load_image(sharp_dir / f.name)
            if sharp.shape != restored.shape:
                sharp = TF.resize(sharp.squeeze(0), restored.shape[-2:]).unsqueeze(0)
            psnr, ssim = psnr_batch(restored, sharp), ssim_batch(restored, sharp)
            psnrs.append(psnr); ssims.append(ssim)
            print(f"{f.name:35s} PSNR={psnr:.2f}dB SSIM={ssim:.4f}")

            if args.save_grid:
                grid = vutils.make_grid([blurred.squeeze(0), restored.squeeze(0), sharp.squeeze(0)],
                                        nrow=3, padding=4, pad_value=1.0)
                save_image(grid, out_dir / f"grid_{f.stem}.png")
        else:
            print(f"{f.name:35s} restored")

    if psnrs:
        print(f"\nMean PSNR: {np.mean(psnrs):.2f} dB  Mean SSIM: {np.mean(ssims):.4f}")


if __name__ == "__main__":
    main()