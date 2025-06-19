# Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals

<center>
Yuxin Lin, Yinglin Zheng, Ming Zeng, Wangzheng Shi

Accepted by ACL 2025

<a href="https://github.com/Linyx1125/MM-F2F"><img src="https://img.shields.io/badge/Code-Github-blue"></a>
<a href="https://arxiv.org/abs/2505.12654"><img src="https://img.shields.io/badge/arXiv-2505.12654-red"></a>
<a href="#Citation"><img src="https://img.shields.io/badge/Citation-BibTeX-green"></a>

</center>

## Abstract

This paper addresses the gap in predicting turn-taking and backchannel actions in human-machine conversations using multi-modal signals (linguistic, acoustic, and visual). To overcome the limitation of existing datasets, we propose an automatic data collection pipeline that allows us to collect and annotate over 210 hours of human conversation videos. From this, we construct a Multi-Modal Face-to-Face (MM-F2F) human conversation dataset, including over 1.5M words and corresponding turn-taking and backchannel annotations from approximately 20M frames. Additionally, we present an end-to-end framework that predicts the probability of turn-taking and backchannel actions from multi-modal signals. The proposed model emphasizes the interrelation between modalities and supports any combination of text, audio, and video inputs, making it adaptable to a variety of realistic scenarios. Our experiments show that our approach achieves state-of-the-art performance on turn-taking and backchannel prediction tasks, achieving a 10% increase in F1-score on turn-taking and a 33% increase on backchannel prediction.

## Get Started

### Environment Setup

Clone the repository:

```bash
git clone https://github.com/Linyx1125/MM-F2F
cd MM-F2F/
```

Setup environment:

```bash
# 1. Create conda environment
conda create -n mm-turn-taking python=3.9
# 2. Activate environment
conda activate mm-turn-taking
# 3. Install pip dependencies
pip install -r requirements.txt
```

### Download Pretrained Model

Download pretrained checkpoint from:

- [[Google Drive]](https://drive.google.com/file/d/1jREeRdQP21jpsqehSj438jJDzBZtMRci/view?usp=sharing)
- [[Baidu Disk]](https://pan.baidu.com/s/1ufEynvShGcH63o755-P6TQ?pwd=1125) (Password: 1125)

### Quick Inference

```bash
python inference.py --input_path "example/input_1.mp4" \
                    --ckpt_path "path/to/ckpt.pt"
```

## Train Custom Model

### Prepare Dataset

To train with our MM-F2F dataset, see [prepare MM-F2F](https://github.com/Linyx1125/MM-F2F/blob/master/dataset/README.md).

### Train uni-modal models:

```bash
python train_unimodal.py  --modal "text" \
                          --data_root <path/to/dataset>

python train_unimodal.py  --modal "audio" \
                          --data_root <path/to/dataset>

python train_unimodal.py  --modal "video" \
                          --data_root <path/to/dataset>
```

### Train end-to-end multi-modal model:

```bash
python  --t_ckpt_path "path/to/t_ckpt.pt" \
        --a_ckpt_path "path/to/a_ckpt.pt" \
        --v_ckpt_path "path/to/v_ckpt.pt" \
        --data_root <path/to/dataset>
```

## Citation

If you find our work helpful or relevant to your research, please consider citing it using the following BibTeX entry and giving a star to the repository.

```
@misc{lin2025predictingturntakingbackchannelhumanmachine,
      title={Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals}, 
      author={Yuxin Lin and Yinglin Zheng and Ming Zeng and Wangzheng Shi},
      year={2025},
      eprint={2505.12654},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.12654}, 
}
```

## Acknowledgement

This repo is built on the excellent work of [batch-face](https://github.com/elliottzheng/batch-face), [whisperx](https://github.com/m-bain/whisperX), and [transformers](https://github.com/huggingface/transformers). We also gratefully borrow some code from [Low-rank-Multimodal-Fusion](https://github.com/Justin1904/Low-rank-Multimodal-Fusion). Many thanks to these great projects!
