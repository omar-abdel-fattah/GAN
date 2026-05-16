"""
Pix2Pix — Full-Image Defect Synthesis (Phase 2)
================================================
Trains a conditional image-to-image GAN (Pix2Pix) on the DeepPCB paired
data: input = clean template image, output = defect image.

Once trained, the generator:
  • Takes any clean template → produces a realistic defective image
  • The annotation (bounding box) comes from the MATCHED real label file,
    so every synthetic image has a guaranteed-correct tight label

This directly attacks the localization gap (mAP50 vs mAP50-95) because the
defects are generated in the exact positions the labels say.

Architecture:
  Generator     : U-Net (encoder-decoder with skip connections) 640×640
  Discriminator : PatchGAN 70×70 patches — classifies overlapping patches
                  as real/fake rather than the whole image

Usage:
    python train_pix2pix.py
    python train_pix2pix.py --epochs 200 --batch 4 --device 0

After training:
    python generate_pix2pix.py --n 1000   (generates 1000 synthetic image+label pairs)
    python merge_and_retrain.py --dataset deeppcb --name pcb_defects_pix2pix
"""

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/wv/omaadeaj_nobackup/downloads/GAN/.ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.utils import save_image
import cv2

GAN_DIR   = Path("/wv/omaadeaj_nobackup/downloads/GAN")
CKPT_DIR  = GAN_DIR / "pix2pix_checkpoints"
IMG_SIZE  = 256   # Pix2Pix works well at 256; 640 needs too much VRAM for batch>1


# ── Paired Dataset ─────────────────────────────────────────────────────────────
class PCBPairDataset(Dataset):
    """
    Returns (template, defective) image pairs from DeepPCB.
    Both resized to IMG_SIZE × IMG_SIZE, normalised to [-1, 1].
    """
    def __init__(self, raw_pcb_dir, split_file, img_size=IMG_SIZE):
        self.pairs   = []
        self.sample_weights = []
        self.img_size = img_size
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),   # grayscale
        ])

        class_counts = {}
        sample_class_sets = []

        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                img_rel, lbl_rel = parts[0], parts[1]
                # img_rel: group20085/20085/20085000.jpg  → actual: 20085000_test.jpg
                stem   = Path(img_rel).stem   # e.g. 20085000
                group  = Path(img_rel).parts[0]
                subdir = Path(img_rel).parts[1]
                test_img = raw_pcb_dir / group / subdir / f"{stem}_test.jpg"
                temp_img = raw_pcb_dir / group / subdir / f"{stem}_temp.jpg"
                lbl_path = raw_pcb_dir / lbl_rel
                if test_img.exists() and temp_img.exists() and lbl_path.exists():
                    self.pairs.append((str(temp_img), str(test_img), str(lbl_path)))

                    # Track classes present in each paired sample for class-balanced sampling.
                    present = set()
                    with open(lbl_path) as lf:
                        for raw in lf:
                            ps = raw.strip().split()
                            if len(ps) < 5:
                                continue
                            cid = int(ps[4]) - 1  # DeepPCB classes are 1..6
                            if 0 <= cid < 6:
                                present.add(cid)
                    if not present:
                        present.add(-1)  # fallback bucket if label is empty
                    sample_class_sets.append(present)
                    for cid in present:
                        class_counts[cid] = class_counts.get(cid, 0) + 1

        # Compute per-sample inverse-frequency weight using classes present in the pair.
        # This oversamples samples containing rare defect classes.
        for present in sample_class_sets:
            invs = [1.0 / max(class_counts[cid], 1) for cid in present]
            self.sample_weights.append(max(invs) if invs else 1.0)

        if self.pairs:
            idx = list(range(len(self.pairs)))
            random.shuffle(idx)
            self.pairs = [self.pairs[i] for i in idx]
            if self.sample_weights:
                self.sample_weights = [self.sample_weights[i] for i in idx]

        print(f"  PCBPairDataset: {len(self.pairs)} pairs from {split_file.name}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        temp_path, test_path, _ = self.pairs[idx]
        temp = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        if temp is None or test is None:
            temp = test = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        temp = cv2.resize(temp, (self.img_size, self.img_size))
        test = cv2.resize(test, (self.img_size, self.img_size))
        return self.tf(temp[:, :, None]), self.tf(test[:, :, None])

    def make_balanced_sampler(self):
        if not self.sample_weights:
            return None
        weights = torch.as_tensor(self.sample_weights, dtype=torch.double)
        return WeightedRandomSampler(weights=weights, num_samples=len(self), replacement=True)


# ── UNet Generator ─────────────────────────────────────────────────────────────
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, bn=True, dropout=False, act="relu"):
        super().__init__()
        layers = []
        if down:
            layers += [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        else:
            layers += [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if bn:
            layers += [nn.BatchNorm2d(out_ch)]
        if dropout:
            layers += [nn.Dropout(0.5)]
        layers += [nn.ReLU(True) if act == "relu" else nn.LeakyReLU(0.2, True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """256×256 UNet: 8 down-blocks, 8 up-blocks with skip connections."""
    def __init__(self, in_ch=1, out_ch=1, nf=64):
        super().__init__()
        # Encoder (no BN on first layer)
        self.d1 = nn.Sequential(nn.Conv2d(in_ch, nf,    4, 2, 1, bias=False), nn.LeakyReLU(0.2, True))
        self.d2 = UNetBlock(nf,    nf*2,  down=True,  bn=True,  act="lrelu")
        self.d3 = UNetBlock(nf*2,  nf*4,  down=True,  bn=True,  act="lrelu")
        self.d4 = UNetBlock(nf*4,  nf*8,  down=True,  bn=True,  act="lrelu")
        self.d5 = UNetBlock(nf*8,  nf*8,  down=True,  bn=True,  act="lrelu")
        self.d6 = UNetBlock(nf*8,  nf*8,  down=True,  bn=True,  act="lrelu")
        self.d7 = UNetBlock(nf*8,  nf*8,  down=True,  bn=True,  act="lrelu")
        self.d8 = nn.Sequential(nn.Conv2d(nf*8, nf*8, 4, 2, 1, bias=False), nn.ReLU(True))  # bottleneck

        # Decoder (skip connections double channels)
        self.u8 = UNetBlock(nf*8,  nf*8,  down=False, bn=True,  dropout=True)
        self.u7 = UNetBlock(nf*16, nf*8,  down=False, bn=True,  dropout=True)
        self.u6 = UNetBlock(nf*16, nf*8,  down=False, bn=True,  dropout=True)
        self.u5 = UNetBlock(nf*16, nf*8,  down=False, bn=True)
        self.u4 = UNetBlock(nf*16, nf*4,  down=False, bn=True)
        self.u3 = UNetBlock(nf*8,  nf*2,  down=False, bn=True)
        self.u2 = UNetBlock(nf*4,  nf,    down=False, bn=True)
        self.u1 = nn.Sequential(
            nn.ConvTranspose2d(nf*2, out_ch, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)
        u  = self.u8(d8);                u = self.u7(torch.cat([u, d7], 1))
        u  = self.u6(torch.cat([u, d6], 1)); u = self.u5(torch.cat([u, d5], 1))
        u  = self.u4(torch.cat([u, d4], 1)); u = self.u3(torch.cat([u, d3], 1))
        u  = self.u2(torch.cat([u, d2], 1)); u = self.u1(torch.cat([u, d1], 1))
        return u


# ── PatchGAN Discriminator ─────────────────────────────────────────────────────
class PatchGANDiscriminator(nn.Module):
    """Classifies 70×70 overlapping patches as real/fake."""
    def __init__(self, in_ch=2, nf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, nf,    4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf,    nf*2,  4, 2, 1, bias=False), nn.BatchNorm2d(nf*2),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*2,  nf*4,  4, 2, 1, bias=False), nn.BatchNorm2d(nf*4),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*4,  nf*8,  4, 1, 1, bias=False), nn.BatchNorm2d(nf*8),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*8,  1,     4, 1, 1, bias=False),
        )

    def forward(self, cond, target):
        return self.net(torch.cat([cond, target], dim=1))


def weights_init(m):
    cls = m.__class__.__name__
    if "Conv" in cls:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in cls:
        nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int, default=200)
    p.add_argument("--batch",     type=int, default=8,
                   help="Batch size. 8 uses ~12GB VRAM at 256px on V100")
    p.add_argument("--lr",        type=float, default=0.0002)
    p.add_argument("--beta1",     type=float, default=0.5,
                   help="Adam beta1")
    p.add_argument("--beta2",     type=float, default=0.999,
                   help="Adam beta2")
    p.add_argument("--lambda-l1", type=float, default=100.0,
                   help="Weight for L1 pixel reconstruction loss")
    p.add_argument("--class-balanced", action="store_true",
                   help="Use class-balanced sampling from paired train set")
    p.add_argument("--img-size",  type=int, default=256)
    p.add_argument("--device",    default="0")
    p.add_argument("--save-every",type=int, default=25)
    p.add_argument("--workers",   type=int, default=4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" else "cpu")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (CKPT_DIR / "samples").mkdir(exist_ok=True)

    raw_pcb = GAN_DIR / "DeepPcb dataset" / "DeepPCB-master" / "PCBData"

    print(f"\n{'='*65}")
    print(f"  Pix2Pix Training  [DeepPCB paired template → defective]")
    print(f"{'='*65}")
    print(f"  Epochs   : {args.epochs}   Batch: {args.batch}   ImgSize: {args.img_size}")
    print(f"  λ_L1     : {args.lambda_l1}   lr: {args.lr}   betas=({args.beta1}, {args.beta2})")
    print(f"  Sampling : {'class-balanced' if args.class_balanced else 'uniform shuffled'}")
    print(f"  Device   : {device}")
    print(f"  Output   : {CKPT_DIR}")
    print(f"{'='*65}\n")

    train_ds = PCBPairDataset(raw_pcb, raw_pcb / "trainval.txt", img_size=args.img_size)
    val_ds   = PCBPairDataset(raw_pcb, raw_pcb / "test.txt",     img_size=args.img_size)

    train_sampler = train_ds.make_balanced_sampler() if args.class_balanced else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader   = DataLoader(val_ds,   batch_size=4,          shuffle=False,
                              num_workers=2, pin_memory=True)

    G = UNetGenerator().to(device)
    D = PatchGANDiscriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Linear LR decay after epoch 100
    def lr_lambda(ep):
        return 1.0 if ep < args.epochs // 2 else 1.0 - (ep - args.epochs // 2) / (args.epochs // 2)
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    bce = nn.BCEWithLogitsLoss()
    l1  = nn.L1Loss()

    # Fixed val pair for visual progress
    fixed_temp, fixed_defect = next(iter(val_loader))
    fixed_temp   = fixed_temp.to(device)
    fixed_defect = fixed_defect.to(device)

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        d_losses, g_losses = [], []

        for temp, defect in train_loader:
            temp   = temp.to(device)
            defect = defect.to(device)

            # ── Train D ────────────────────────────────────────────────────────
            opt_D.zero_grad()
            fake    = G(temp)
            real_pred = D(temp, defect)
            fake_pred = D(temp, fake.detach())
            loss_D = 0.5 * (bce(real_pred, torch.ones_like(real_pred)) +
                            bce(fake_pred, torch.zeros_like(fake_pred)))
            loss_D.backward()
            opt_D.step()

            # ── Train G ────────────────────────────────────────────────────────
            opt_G.zero_grad()
            fake      = G(temp)
            fake_pred = D(temp, fake)
            loss_G_adv = bce(fake_pred, torch.ones_like(fake_pred))
            loss_G_l1  = l1(fake, defect) * args.lambda_l1
            loss_G     = loss_G_adv + loss_G_l1
            loss_G.backward()
            opt_G.step()

            d_losses.append(loss_D.item())
            g_losses.append(loss_G.item())

        sched_G.step(); sched_D.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [ep {epoch:4d}/{args.epochs}]  lossD={np.mean(d_losses):.4f}  "
                  f"lossG={np.mean(g_losses):.4f}  lr={opt_G.param_groups[0]['lr']:.6f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            G.eval()
            with torch.no_grad():
                fake_val = G(fixed_temp)
            grid = torch.cat([fixed_temp, fake_val, fixed_defect], dim=3)  # side-by-side
            grid = (grid + 1) / 2
            save_image(grid, CKPT_DIR / "samples" / f"epoch_{epoch:04d}.png", nrow=1)
            torch.save(G.state_dict(), CKPT_DIR / "generator_final.pt")
            torch.save(D.state_dict(), CKPT_DIR / "discriminator_final.pt")
            torch.save({"epoch": epoch, "opt_G": opt_G.state_dict(),
                        "opt_D": opt_D.state_dict()},
                       CKPT_DIR / "optimizer_state.pt")

    print(f"\n  Pix2Pix training complete → {CKPT_DIR}")
    print(f"\n  Next step:")
    print(f"    python generate_pix2pix.py --n 1000")
    print(f"    python merge_and_retrain.py --dataset deeppcb --name pcb_defects_pix2pix\n")


if __name__ == "__main__":
    main()
