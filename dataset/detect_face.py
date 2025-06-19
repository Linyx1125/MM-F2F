import os
import glob
import threading

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str)
parser.add_argument("--csv_root", type=str, default="./")
parser.add_argument("--raw_root", type=str, default="./raw/")
parser.add_argument("--out_root", type=str, default="./video/")
parser.add_argument("--log_path", type=str, default="log.txt")
parser.add_argument("--num_threading", type=int, default=4)
args = parser.parse_args()

split = args.split
assert split in ["train", "val", "test"]
df = pd.read_csv(os.path.join(args.csv_root, f"{split}.csv"))
raw_root = args.raw_root
out_root = args.out_root
num_threading = args.num_threading

# === face detector ===
batch_size = 16
THRESHOLD = 0.99
from batch_face import RetinaFace
detector = RetinaFace(gpu_id=0)

def pred_batch(batch_frames, batch_outpaths, batch_labels):
    all_faces = detector(batch_frames)
    # all_bbox  = []

    for (faces, frame, outpath, label) in zip(all_faces, batch_frames, batch_outpaths, batch_labels):
        valid_faces = list(filter(lambda face: face[2] > THRESHOLD, faces))
        sorted_faces = sorted(valid_faces, key=lambda face: face[0][0])

        if len(sorted_faces) < 2:
            continue
        if label == 0:
            target_face = sorted_faces[0]
        elif label == 1:
            target_face = sorted_faces[-1]
        else:
            continue

        # all_bbox.append(target_face[0])
        bbox = target_face[0]
        x1, y1, x2, y2 = np.round(bbox).astype(int)
        try:
            face_frame = frame[y1:y2, x1:x2, :]
            cv2.imwrite(outpath, face_frame)
        except:
            continue


# === load video paths ===
def load_video_paths():
    video_name_pattern = "*.mp4"
    video_path_dict = {}
    video_paths = glob.glob(os.path.join(raw_root, video_name_pattern))
    for video_path in video_paths:
        vid = os.path.basename(video_path).replace(video_name_pattern, "")
        video_path_dict[vid] = video_path
    return video_path_dict
video_path_dict = load_video_paths()
# video_path_dict = {
#     "0NbUgJWamig": "/data/turn_taking/final/0NbUgJWamig_25.mp4",
# }
print(f"Loaded {len(video_path_dict.keys())} video paths")


def process(video_ids):
    t_bar = tqdm(video_ids)
    for video_id in t_bar:
        t_bar.set_description(f"Processing {video_id}")
        if video_id not in video_path_dict:
            continue
        raw_path = video_path_dict[video_id]
        # cap = cv2.VideoCapture(raw_path)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        cap = None
        fps = None

        out_folder = os.path.join(out_root, video_id)
        os.makedirs(out_folder, exist_ok=True)

        sub_df = df[df["video_id"] == video_id]
        for i, row in sub_df.iterrows():
            try:
                video_id, sentence_id, text, label, start, end, speaker = row
                out_clip_folder = os.path.join(out_folder, f"{sentence_id}")
                if os.path.exists(out_clip_folder):
                    if len(os.listdir(out_clip_folder)) == batch_size:
                        continue
                os.makedirs(out_clip_folder, exist_ok=True)

                if cap is None:
                    cap = cv2.VideoCapture(raw_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)

                end_frame = int(end * fps)
                start_frame = end_frame - batch_size
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                batch_frames, batch_outpaths, batch_labels = [], [], []
                for i_batch in range(batch_size):
                    out_path = os.path.join(out_clip_folder, f"{i_batch}.jpg")
                    if os.path.exists(out_path): continue
                    cur_frame = cap.read()[1]
                    if cur_frame is None: break
                    batch_frames.append(cur_frame)
                    batch_outpaths.append(out_path)
                    batch_labels.append(speaker)
                if len(batch_labels) > 0:
                    pred_batch(batch_frames, batch_outpaths, batch_labels)
            except Exception as e:
                with open(args.log_path, "a") as f:
                    f.write(f"Error in {video_id} {sentence_id}\n")
                continue


if __name__ == "__main__":
    video_ids = list(sorted(set(df["video_id"])))

    len_video_ids = len(video_ids)

    for i in range(num_threading):
        threading.Thread(target=process, args=(video_ids[i::num_threading],)).start()