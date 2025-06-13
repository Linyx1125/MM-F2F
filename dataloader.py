import os, glob
import pandas as pd
import librosa
import cv2
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor

from datasets import Dataset
from torch.utils.data import DataLoader


tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

class MultiModalDataset(Dataset):
    def __init__(self, data_root, split, modal="all", tokenizer=None, audio_processor=None, video_processor=None):
        assert split in ["train", "val", "test"]
        assert modal in ["audio", "video", "text", "all"]
        self.modal = modal

        df = pd.read_csv(os.path.join(data_root, f"{split}.csv"))
        sid = df["sentence_id"].values
        text = df["text"].values
        label = df["label"].values
        assert len(sid) == len(label)

        if self.modal in ["text", "all"]:
            assert tokenizer is not None
            self.tokenizer = tokenizer

        if self.modal in ["audio", "all"]:
            assert audio_processor is not None
            audio_root = os.path.join(data_root, "audio")
            self.audio_processor = audio_processor
            self.sampling_rate = self.audio_processor.feature_extractor.sampling_rate

        if self.modal in ["video", "all"]:
            assert video_processor is not None
            video_root = os.path.join(data_root, "video")
            self.video_processor = video_processor
            self.n_imgs = 16

        self.data_list = []
        for (_sid, _text, _label) in tqdm(zip(sid, text, label), total=len(sid)):
            data = { "label": _label }
            video_id = "_".join(_sid.split("_")[:-2])
            if self.modal in ["text", "all"]:
                text_ids = self.tokenizer(_text)
                data["text"] = text_ids
            if self.modal in ["audio", "all"]:
                audio_path = os.path.join(audio_root, video_id, f"{_sid}.mp3")
                data["audio"] = audio_path
            if self.modal in ["video", "all"]:
                img_paths = []
                for i in range(self.n_imgs):
                    img_path = os.path.join(video_root, video_id, _sid, f"{i}.jpg")
                    if not os.path.exists(img_path): break
                    img_paths.append(img_path)
                if len(img_paths) < self.n_imgs: continue
                data["video"] = img_paths
            self.data_list.append(data)


    def load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sampling_rate)
        X = self.audio_processor(audio, sampling_rate=self.sampling_rate, return_tensors="pt").input_values
        return X
    
    def load_images(self, paths):
        imgs = [cv2.imread(path)[:,:,::-1] for path in paths]
        X = self.video_processor(imgs, return_tensors="pt")
        return X

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        out_dict = { "label": data["label"] }
        if "text" in data:
            out_dict["text"] = data["text"]
        if "audio" in data:
            out_dict["audio"] = self.load_audio(data["audio"])
        if "video" in data:
            out_dict["video"] = self.load_images(data["video"])
        return out_dict
    

def collate_fn(batch):
    out_tuple = ()
    if "text" in batch[0]:
        text = {
            "input_ids": torch.tensor([x["text"]["input_ids"] for x in batch]),
            "attention_mask": torch.tensor([x["text"]["attention_mask"] for x in batch]),
        }
        out_tuple += (text,)
    if "audio" in batch[0]:
        audio = torch.stack([x["audio"][0] for x in batch])
        out_tuple += (audio,)
    if "video" in batch[0]:
        video = {
            "pixel_values": torch.stack([x["video"]["pixel_values"][0] for x in batch]),
        }
        out_tuple += (video,)
    label = torch.tensor([x["label"] for x in batch])
    out_tuple += (label,)
    return out_tuple


if __name__ == "__main__":
    data_root = "./dataset"

    test_set = MultiModalDataset(data_root, "test", modal="all", tokenizer=tokenizer, audio_processor=audio_processor, video_processor=video_processor)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    text, audio, video, label = next(iter(test_loader))

    print(text["input_ids"].shape)
    print(audio.shape)
    print(video["pixel_values"].shape)
    print(label)