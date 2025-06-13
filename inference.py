import os

import cv2
import numpy as np
import torch
from torch import nn

import librosa

import whisperx

from model.mm import LanguageAudioVisionModel, load_processors
from utils import detect_faces

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--device', type=str, default="cuda")
# video args
parser.add_argument('--n_imgs', type=int, default=16)
# audio args
parser.add_argument('--samplerate', type=int, default=16000)
parser.add_argument('--max_audio_length', type=int, default=400000)
args = parser.parse_args()


class WhisperXManager:
    def __init__(self, args, language="en"):
        self.args = args
        self.model = whisperx.load_model(
            "large-v3", 
            self.args.device, 
            compute_type="float16", 
            language=language,
            asr_options={
                "suppress_numerals": True,
            },
        )
        self.model_a, self.metadata = whisperx.load_align_model(language_code=language, device=self.args.device)

    def transcribe(self, audio, return_words=False):
        result = self.model.transcribe(audio, batch_size=16, chunk_size=10)["segments"]
        if return_words:
            result = whisperx.align(result, self.model_a, self.metadata, audio, self.args.device, return_char_alignments=False)
            return result["word_segments"]
        return result


class TurnTakingManager:
    def __init__(self, args):
        self.args = args
        self.tokenizer, self.text_processor, self.audio_processor, self.video_processor = load_processors()
        self.sampling_rate = self.audio_processor.feature_extractor.sampling_rate
        
        self.model = LanguageAudioVisionModel().to(self.args.device)
        self.model.load_state_dict(torch.load(self.args.ckpt_path), strict=False)
        self.model.eval()

        self.whisperx = WhisperXManager(self.args)

    def load_text(self, text):
        if isinstance(text, str):
            text = self.text_processor(text)
            if text == "": return None
            text = self.tokenizer(text, return_tensors="pt").input_ids
        return text

    def load_audio(self, audio):
        if isinstance(audio, str) and os.path.exists(audio):
            audio, _ = librosa.load(audio, sr=self.sampling_rate)
        return audio
    
    def load_video(self, video):
        if isinstance(video, str) and os.path.exists(video):
            frames = []
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)
            return frames, fps
        return None, None
    
    def transcribe(self, audio, return_words=False):
        return self.whisperx.transcribe(audio, return_words)

    def predict(self, text, audio, frames):
        text_input = text
        audio_input = self.audio_processor(audio, sampling_rate=self.sampling_rate, return_tensors="pt").input_values
        face_frames = detect_faces(frames)
        if face_frames is None:
            return None
        video_input = self.video_processor(face_frames).pixel_values

        text_input = torch.tensor(text_input).to(self.args.device)
        audio_input = audio_input.to(self.args.device)
        video_input = torch.tensor(video_input).to(self.args.device)

        with torch.no_grad():
            preds = self.model(
                text_input, 
                audio_input, 
                video_input,
            )
            preds = nn.Sigmoid()(preds).cpu().detach().numpy()
        
        logit = preds[0]
        normalized_logit = logit / logit.sum()
        return normalized_logit


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vframes = []
    while cap.isOpened():
        ret, vframe = cap.read()
        if not ret:
            break
        vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
        vframes.append(vframe)
    cap.release()
    return vframes, fps


def read_audio(audio_path, samplerete):
    aframes, _ = librosa.load(audio_path, sr=samplerete)
    return aframes


def idx2label(idx):
    return {0: "keep", 1: "turn-taking", 2: "backchannel"}[idx]


def logit2action(logit):
    action = logit.argmax(0).item()
    return action


def inference(manager, aframes, vframes, fps):
    asr_res = manager.transcribe(aframes, return_words=True)
    if len(asr_res) == 0:
        return None
    word_segments = asr_res

    text_seq = ""
    for i, seg in enumerate(word_segments):
        try:
            start = seg["start"]
            end  = seg["end"]

            text_seq += (seg["word"] + " ")
            text_input = manager.load_text(text_seq)
            audio_input = aframes[:int(end * args.samplerate)][-args.max_audio_length:]
            frame_end_idx = int(end * fps)
            if frame_end_idx < args.n_imgs:
                continue
            video_input = vframes[frame_end_idx - args.n_imgs:frame_end_idx]
            if len(video_input) < args.n_imgs:
                continue

            logit = manager.predict(text_input, audio_input, video_input)
            action = logit2action(logit)

            print("========================================")
            print("INPUT:", text_seq)
            print("ACTION:", idx2label(action), logit)
            print("========================================")

        except:
            pass


def main():
    # load video and audio
    vframes, fps = read_video(args.input_path)
    aframes = read_audio(args.input_path, args.samplerate)

    # load model
    manager = TurnTakingManager(args)

    inference(manager, aframes, vframes, fps)


if __name__ == "__main__":
    main()