import os
import time
import random
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from datasets import Dataset
from torch.utils.data import DataLoader

from model.mm import LanguageAudioVisionModel, load_processors
from dataloader import MultiModalDataset, collate_fn

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--t_ckpt_path", type=str, required=True)
parser.add_argument("--a_ckpt_path", type=str, required=True)
parser.add_argument("--v_ckpt_path", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
# data params
parser.add_argument("--data_root", type=str, default="dataset/")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--n_workers", type=int, default=4)
# train params
parser.add_argument("--n_epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--random_drop_modal_rate", type=float, default=0.05)
# training log
parser.add_argument("--log_dir", type=str, default="log")
parser.add_argument("--model_name", type=str, default="_")
args = parser.parse_args()


def init_log_dir():
    os.makedirs(args.log_dir, exist_ok=True)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).replace(" ", "_")
    task_dir = os.path.join(args.log_dir, f"{time_str}{args.model_name}")
    writer = SummaryWriter(log_dir=task_dir)
    return writer, task_dir


def cal_metric(all_labels, all_logits):
    accuracy = accuracy_score(all_labels, all_logits)
    recall = recall_score(all_labels, all_logits, average=None)
    f1 = f1_score(all_labels, all_logits, average=None)
    precision = precision_score(all_labels, all_logits, average=None)
    return accuracy, recall, f1, precision


def idx2label(idx):
    return ["keep", "turn", "bc"][idx]


def main():
    tokenizer, _, audio_processor, video_processor = load_processors()
    train_set = MultiModalDataset(
        data_root=args.data_root,
        split="train",
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        video_processor=video_processor,
    )
    val_set = MultiModalDataset(
        data_root=args.data_root,
        split="val",
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        video_processor=video_processor,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)

    model = LanguageAudioVisionModel().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    writer, task_dir = init_log_dir()

    n_epoch = 10
    random_drop_modal_rate = 0.05
    t_bar = tqdm(range(n_epoch))
    for epoch in t_bar:
        # train
        model.train()
        n_batch = len(train_loader)
        epoch_loss = 0.0
        all_labels, all_logits = [], []
        for i, (text, audio, vision, y) in enumerate(train_loader):
            if audio is None:
                continue
            text = text["input_ids"].to("cuda")
            audio = audio.to("cuda")
            vision = vision["pixel_values"].to("cuda")
            y = y.to("cuda")
            optimizer.zero_grad()
            
            if random.random() < random_drop_modal_rate:
                drop_modal = random.choice(["text", "audio", "vision"])
                if drop_modal == "text":
                    text = None
                elif drop_modal == "audio":
                    audio = None
                elif drop_modal == "vision":
                    vision = None
                # print("drop modal:", drop_modal)
            pred = model(text, audio, vision)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            all_labels.extend(y.detach().cpu().numpy())
            all_logits.extend(pred.argmax(dim=1).detach().cpu().numpy())

            t_bar.set_description(f"Epoch {epoch} training | Batch {i}/{n_batch} | Loss {loss.item():.4f}")
            epoch_loss += (loss.item() * args.batch_size)
        epoch_loss = epoch_loss / n_batch
        writer.add_scalar("train/loss", epoch_loss, epoch)
        accuracy, recall, f1, precision = cal_metric(np.array(all_labels), np.array(all_logits))
        writer.add_scalar("train/accuracy", accuracy, epoch)
        for idx, r in enumerate(recall):
            writer.add_scalar(f"train/{idx2label(idx)}_recall", r, epoch)
        for idx, f in enumerate(f1):
            writer.add_scalar(f"train/{idx2label(idx)}_f1", f, epoch)
        for idx, p in enumerate(precision):
            writer.add_scalar(f"train/{idx2label(idx)}_precision", p, epoch)

        # val
        model.eval()
        n_batch = len(val_loader)
        val_loss = 0.0
        all_labels, all_logits = [], []
        for i, (text, audio, vision, y) in enumerate(val_loader):
            if audio is None:
                continue
            text = text["input_ids"].to("cuda")
            audio = audio.to("cuda")
            vision = vision["pixel_values"].to("cuda")
            y = y.to("cuda")
            with torch.no_grad():
                pred = model(text, audio, vision)
                loss = criterion(pred, y)
                
                all_labels.extend(y.detach().cpu().numpy())
                all_logits.extend(pred.argmax(dim=1).detach().cpu().numpy())
                val_loss += (loss.item() * args.batch_size)
        val_loss = val_loss / n_batch
        writer.add_scalar("val/loss", val_loss, epoch)
        accuracy, recall, f1, precision = cal_metric(np.array(all_labels), np.array(all_logits))
        writer.add_scalar("val/accuracy", accuracy, epoch)
        for idx, r in enumerate(recall):
            writer.add_scalar(f"val/{idx2label(idx)}_recall", r, epoch)
        for idx, f in enumerate(f1):
            writer.add_scalar(f"val/{idx2label(idx)}_f1", f, epoch)
        for idx, p in enumerate(precision):
            writer.add_scalar(f"val/{idx2label(idx)}_precision", p, epoch)

        save_path = os.path.join(task_dir, f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()