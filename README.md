***Efficient Large-Scale Traffic Forecasting with Transformers: A Spatial Data Management Perspective***

<img width="347" alt="sketch" src="https://github.com/user-attachments/assets/fcda0be0-d4b3-45c7-a919-ef9b0f42bf98" />

🙋 Please let us know if you find out a mistake or have any suggestions!

🌟 If you find this resource helpful, please consider to star this repository and cite our research:

@article{fang2024efficient,
  title={Efficient Large-Scale Traffic Forecasting with Transformers: A Spatial Data Management Perspective},
  author={Fang, Yuchen and Liang, Yuxuan and Hui, Bo and Shao, Zezhi and Deng, Liwei and Liu, Xu and Jiang, Xinke and Zheng, Kai},
  journal={arXiv preprint arXiv:2412.09972},
  year={2024}
}
**Introduction**
PatchSTG is an attention-based dynamic spatial modeling method that uses irregular spatial patching for efficient large-scale traffic forecasting. Notably, we show that spatiotemporal graphs can be patched on the spatial dimension, effectively reducing complexity in attention.

<img width="348" alt="frame" src="https://github.com/user-attachments/assets/2cbaea3a-9b07-4ecb-8fd0-c42eec1df2a9" />


PatchSTG comprises four components: (1) embedding the input traffic into high-dimensional representations with spatio-temporal properties, (2) segmenting the large-scale input into balanced and non-overlapped patches on the spatial dimension with irregularly distributed points, (3) using depth and breadth attentions on the patched input to capture local and global spatial dependencies efficiently, and (4) projecting representations to the predicted future traffic.
<img width="704" alt="patching" src="https://github.com/user-attachments/assets/fb5685e9-f5bf-4fe8-b7c9-1af624d32da7" />


**Requirements**
torch==1.11.0
timm==1.0.12
scikit_learn==1.0.2
tqdm==4.67.1
pandas==1.4.1
numpy==1.22.3


**Folder Structure**
└── code-and-data
    ├── config                 # Including detail configurations
    ├── cpt                    # Storing pre-trained weight files (manually create the folder and download files)
    ├── data                   # Including traffic data (download), adj files (generated), and the meta data
    ├── lib
    │   |──  utils.py          # Codes of preprocessing datasets and calculating metrics
    ├── log                    # Storing log files
    ├── model
    │   |──  models.py         # The core source code of our PatchSTG
    ├── main.py                # This is the main file for training and testing
    └── README.md              # This document
**Datasets**
You can access the well pre-processed datasets from [https://drive.google.com/drive/u/1/folders/1a3R5rFST4O0Ohr4QCXhsIKkezWLriE-K], then place the downloaded contents under the correspond dataset folder such as ./data/SD.

**PreTrained Weights**
You can access the pretrained weights from [https://drive.google.com/drive/u/1/folders/1NJW-fJgrN8o-9VTsSrEzER150X3EFGPG], then place the downloaded contents under the constructed cpt folder ./cpt.

**Quick Demos**
Download datasets and place them under ./data
We provide pre-trained weights of results in the paper and the detail configurations under the folder ./config. For example, you can test the SD dataset by:
python main.py --config ./config/SD.conf
If you want to train the model yourself, you can use the code at line 262 of the main file.

Authors: Yuchen Fang, Yanjun Qin, Haiyong Luo, Fang Zhao, Bingbing Xu, Liang Zeng, Chenxing Wang.
