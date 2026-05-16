"""
Class-Conditional DCGAN — Patch-Level Defect Synthesis (Phase 1)
=================================================================
Trains a conditional DCGAN on the extracted 64×64 defect patches.
The generator learns to synthesise new defect patches conditioned on
class label → lets us oversample rare classes cheaply.

Architecture:
  Generator  : noise(100) + class_embed(32) → 4×4 → 8×8 → 16×16 → 32×32 → 64×64
  Discriminator: image(1 or 3ch) + class_embed(spatial) → patch real/fake

Output:
  gan_checkpoints/<dataset>/
      generator_final.pt
      discriminator_final.pt
      samples/epoch_NNN/  (8×8 grid per class)

Usage:
    python train_gan.py --dataset deeppcb
    python train_gan.py --dataset deeppcb --classes 0 5     # only specific classes
    python train_gan.py --dataset deeppcb --epochs 300 --batch 128
"""

import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/wv/omaadeaj_nobackup/downloads/GAN/.ultralytics")

import argparse
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import cv2

GAN_DIR       = Path("/wv/omaadeaj_nobackup/downloads/GAN")
PATCH_ROOT    = GAN_DIR / "gan_patches"
CKPT_ROOT     = GAN_DIR / "gan_checkpoints"
CLASS_NAMES   = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
NC            = len(CLASS_NAMES)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
NZ         = 100   # latent noise dim
EMBED_DIM  = 32    # class embedding dim
NGF        = 64    # generator feature maps
NDF        = 64    # discriminator feature maps
IMG_SIZE   = 64    # patch size


def add_instance_noise(x, sigma):
    if sigma <= 0:
        return x
    return (x + sigma * torch.randn_like(x)).clamp(-1, 1)


# ── Dataset ───────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, patch_dir, class_ids, channels=1):
        self.samples   = []
        self.channels  = channels
        tf = [
            transforms.ToTensor(),
            transforms.Normalize([0.5]*channels, [0.5]*channels),
        ]
        self.transform = transforms.Compose(tf)

        for cls_id in class_ids:
            cls_dir = patch_dir / f"class_{cls_id}_{CLASS_NAMES[cls_id]}"
            if not cls_dir.exists():
                continue
            for p in cls_dir.glob("*.png"):
                self.samples.append((str(p), cls_id))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls_id = self.samples[idx]
        flag = cv2.IMREAD_GRAYSCALE if self.channels == 1 else cv2.IMREAD_COLOR
        img  = cv2.imread(path, flag)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if self.channels == 1:
            img = img[:, :, None]  # H×W×1
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img)
        return tensor, cls_id


# ── Generator ─────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, nz, embed_dim, ngf, nc_out):
        super().__init__()
        self.embed = nn.Embedding(NC, embed_dim)
        in_dim = nz + embed_dim
        self.net = nn.Sequential(
            # 1×1
            nn.ConvTranspose2d(in_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            # 4×4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            # 8×8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            # 16×16
            nn.ConvTranspose2d(ngf*2, ngf,   4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),   nn.ReLU(True),
            # 32×32
            nn.ConvTranspose2d(ngf,   nc_out, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 64×64
        )

    def forward(self, z, labels):
        emb = self.embed(labels).unsqueeze(-1).unsqueeze(-1)  # B×E×1×1
        x   = torch.cat([z, emb], dim=1)
        return self.net(x)


# ── Discriminator ─────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self, embed_dim, ndf, nc_in):
        super().__init__()
        self.embed = nn.Embedding(NC, embed_dim)
        # Project embedding to spatial map added as extra channels
        self.embed_proj = nn.Linear(embed_dim, IMG_SIZE * IMG_SIZE)
        in_ch = nc_in + 1  # image channels + label channel
        self.net = nn.Sequential(
            # 64×64
            nn.Conv2d(in_ch, ndf,    4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # 32×32
            nn.Conv2d(ndf,   ndf*2,  4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            # 16×16
            nn.Conv2d(ndf*2, ndf*4,  4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            # 8×8
            nn.Conv2d(ndf*4, ndf*8,  4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, True),
            # 4×4
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            # 1×1
        )

    def forward(self, img, labels):
        B = img.size(0)
        emb = self.embed(labels)                                   # B×E
        lmap = self.embed_proj(emb).view(B, 1, IMG_SIZE, IMG_SIZE) # B×1×64×64
        x = torch.cat([img, lmap], dim=1)
        return self.net(x).view(B)


def weights_init(m):
    cls = m.__class__.__name__
    if "Conv" in cls:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in cls:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  choices=["deeppcb", "pku"], default="deeppcb")
    p.add_argument("--classes",  type=int, nargs="+", default=list(range(NC)),
                   help="Class IDs to include (default: all)")
    p.add_argument("--epochs",   type=int, default=300)
    p.add_argument("--batch",    type=int, default=128)
    p.add_argument("--lr",       type=float, default=0.0002)
    p.add_argument("--channels", type=int, default=1, choices=[1, 3],
                   help="1=grayscale (DeepPCB is binary), 3=RGB (PKU)")
    p.add_argument("--device",   default="0")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--workers",  type=int, default=4)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--gan-loss", choices=["bce", "hinge"], default="hinge",
                   help="GAN objective. hinge is generally more stable than BCE for this setup")
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Only used for BCE. Real target becomes (1 - label_smoothing)")
    p.add_argument("--instance-noise", type=float, default=0.08,
                   help="Gaussian noise std added to real/fake images, linearly decayed to 0")
    p.add_argument("--d-updates", type=int, default=1,
                   help="Number of discriminator updates per batch")
    p.add_argument("--g-updates", type=int, default=1,
                   help="Number of generator updates per batch")
    return p.parse_args()


def main():
    args   = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" else "cpu")

    patch_dir = PATCH_ROOT / args.dataset
    ckpt_dir  = CKPT_ROOT  / args.dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "samples").mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  cDCGAN Training  [{args.dataset.upper()}]")
    print(f"{'='*60}")
    print(f"  Patches  : {patch_dir}")
    print(f"  Epochs   : {args.epochs}  Batch: {args.batch}")
    print(f"  Device   : {device}")
    print(f"  Classes  : {[CLASS_NAMES[i] for i in args.classes]}")
    print(f"  Loss     : {args.gan_loss}   D/G updates: {args.d_updates}/{args.g_updates}")
    print(f"  Noise    : {args.instance_noise} -> 0 (linear decay)")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = PatchDataset(patch_dir, args.classes, channels=args.channels)
    if len(dataset) == 0:
        raise RuntimeError(f"No patches found in {patch_dir}. Run extract_patches.py first.")
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                         num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"  Total patches in training set: {len(dataset)}\n")

    # ── Models ────────────────────────────────────────────────────────────────
    G = Generator(NZ, EMBED_DIM, NGF, args.channels).to(device)
    D = Discriminator(EMBED_DIM, NDF, args.channels).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    # Fixed noise for sample grids - only for the classes being trained.
    fixed_classes = list(args.classes)
    fixed_z    = torch.randn(len(fixed_classes) * 8, NZ, 1, 1, device=device)
    fixed_lbls = torch.tensor([c for c in fixed_classes for _ in range(8)], device=device)

    history = []

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        d_losses, g_losses = [], []
        sigma = max(0.0, args.instance_noise * (1.0 - (epoch - 1) / max(1, args.epochs - 1)))

        for real, labels in loader:
            real   = real.to(device)
            labels = labels.to(device)
            B      = real.size(0)
            real_t = torch.full((B,), 1.0 - args.label_smoothing, device=device)
            fake_t = torch.zeros(B, device=device)

            # ── Train D ───────────────────────────────────────────────────────
            for _ in range(args.d_updates):
                opt_D.zero_grad()
                z = torch.randn(B, NZ, 1, 1, device=device)
                fake = G(z, labels).detach()

                real_in = add_instance_noise(real, sigma)
                fake_in = add_instance_noise(fake, sigma)

                d_real = D(real_in, labels)
                d_fake = D(fake_in, labels)

                if args.gan_loss == "hinge":
                    loss_D = 0.5 * (torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean())
                else:
                    loss_real = criterion(d_real, real_t)
                    loss_fake = criterion(d_fake, fake_t)
                    loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()
                opt_D.step()

            # ── Train G ───────────────────────────────────────────────────────
            for _ in range(args.g_updates):
                opt_G.zero_grad()
                z = torch.randn(B, NZ, 1, 1, device=device)
                fake = G(z, labels)
                fake_in = add_instance_noise(fake, sigma)
                d_fake = D(fake_in, labels)

                if args.gan_loss == "hinge":
                    loss_G = -d_fake.mean()
                else:
                    loss_G = criterion(d_fake, torch.ones(B, device=device))

                loss_G.backward()
                opt_G.step()

            d_losses.append(loss_D.item())
            g_losses.append(loss_G.item())

        d_mean = np.mean(d_losses)
        g_mean = np.mean(g_losses)
        history.append({"epoch": epoch, "lossD": d_mean, "lossG": g_mean})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [ep {epoch:4d}/{args.epochs}]  lossD={d_mean:.4f}  lossG={g_mean:.4f}  sigma={sigma:.4f}")

        # ── Save sample grid ──────────────────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_z, fixed_lbls)  # (classes*8) × C × 64 × 64
            # Denorm to [0,1]
            samples = (samples + 1) / 2
            sample_dir = ckpt_dir / "samples" / f"epoch_{epoch:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for i, cls_id in enumerate(fixed_classes):
                grid = samples[i*8:(i+1)*8]
                save_image(grid, sample_dir / f"class_{cls_id}_{CLASS_NAMES[cls_id]}.png",
                           nrow=8, normalize=False)

            # Save checkpoints: both rolling final files and per-epoch bundles.
            torch.save(G.state_dict(), ckpt_dir / "generator_final.pt")
            torch.save(D.state_dict(), ckpt_dir / "discriminator_final.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "args": vars(args),
                    "generator": G.state_dict(),
                    "discriminator": D.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_D": opt_D.state_dict(),
                    "lossD": float(d_mean),
                    "lossG": float(g_mean),
                },
                ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt",
            )

    # Save history
    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Training complete. Checkpoints → {ckpt_dir}")
    print(f"\n  Next step:")
    print(f"    python generate_synthetic.py --dataset {args.dataset}\n")


if __name__ == "__main__":
    main()
