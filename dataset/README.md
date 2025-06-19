# Prepare MM-F2F Dataset

**Note**: To respect the privacy of the original video uploaders, we have decided not to release the processed data directly for now. Instead, we provide the original video links along with our data processing scripts.

## Data Annotation

Download our data annotation from:

- [[Google Drive]](https://drive.google.com/drive/folders/1EJ-GJOqv2E2e5FFQwCRt4tdqvyaD-zZ5?usp=sharing)
- [[Baidu Disk]](https://pan.baidu.com/s/1M5TL8Tc8XmXfHMzH3VOZkg?pwd=1125) (Password: 1125)

And unzip. The data structure is as follows:

``` bash
<data/root>
  ├─ preprocess/
    ├─ train.csv
    ├─ val.csv
    └─ test.csv
  └─ word_level_split/
    └─ <video_id>.csv
```

The `preprocess/` directory contains annotation files of each split. In each file, the videos are word-level annotated by:

- `video_id`: video ID
- `sentence_id`: sentence ID
- `text`: text transcription
- `start`: start time (seconds) of current sentence
- `end`: end time (seconds) of current sentence
- `label`: action of the listener after current utterance, `0` means keeping the speaker talking, `1` means turn-taking, `2` means providing a backchannel
- `speaker`: we guarantee that each clip consists of 2 person, `0` means the left one is speaking while `1` means the right one is speaking

## Prepare Data

### Download videos

Use a third-party tool to download videos from YouTube by `video_id`.

### Segment audio clips

For each sentence, crop the audio from the corresponding video using the `start` and `end` time. Save the clips in the following structure:

``` bash
<path/to/dataset>
  ├─ audio
    ├─ <video_id>
      ├─ <sentence_id>.mp3
      ├─ <sentence_id>.mp3
      └─ ...
    ├─ <video_id>
      ├─ <sentence_id>.mp3
      ├─ <sentence_id>.mp3
    └─ ...
  ├─ train.csv
  ├─ val.csv
  └─ test.csv
```

### Extract face frames

For each sentence, use a face detection tool (e.g., [batch-face](https://github.com/elliottzheng/batch-face)) to detect the active `speaker`. You may choose to only keep cropped and aligned faces for efficiency. Store them as:

``` bash
<path/to/dataset>
  ├─ video
    ├─ <video_id>
      ├─ <sentence_id>
        ├─ 0.jpg
        ├─ 1.jpg
        └─ ...
    ├─ <video_id>
      ├─ <sentence_id>
        ├─ 0.jpg
        ├─ 1.jpg
        └─ ...
    └─ ...
  ├─ train.csv
  ├─ val.csv
  └─ test.csv
```

You can refer to or directly use our [detect_face.py](https://github.com/Linyx1125/MM-F2F/blob/master/dataset/detect_face.py) script for face extraction.

> 📌 Note: If you're solely reproducing the baseline in our paper, it's sufficient to store the last 16 frames' faces for each sentence.

### Finally,

the data structure is as follows:

``` bash
<path/to/dataset>
  ├─ audio
    ├─ <video_id>
      ├─ <sentence_id>.mp3
      └─ ...
    └─ ...
  ├─ video
    ├─ <video_id>
      ├─ <sentence_id>
        ├─ 0.jpg
        └─ ...
      └─ ...
    └─ ...
  ├─ train.csv
  ├─ val.csv
  └─ test.csv
```