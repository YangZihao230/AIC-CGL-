# [TIP 2025] Universal Fine-Grained Visual Categorization by Concept Guided Learning

This is the official implementation of our work entitled as ```Universal Fine-Grained Visual Categorization by Concept Guided Learning```, which has been accepted by ```IEEE Transactions on Image Processing (TIP'2025)```.

## Fine-grained Land-Cover Dataset (FGLCD) Download

- For ```GoogleDrive``` user, please download by

- For ```BaiduDrive``` user, please download by

## Relation to the Existing Million-AID dataset 

*[Million-AID dataset](https://captain-whu.github.io/DiRS/)* has about one million aerial scene samples from high-resolution satellite images from a variety of imaging sensors (e.g., worldview-2, Gaofen-2, and etc.) in total, but the weakness includes:

- Most of the samples in million-AID are annotated automatically or semi-automatically, i.e., without human-level supervision.

- Only about 10,000 samples have the publicly-available ground truth, which poses a bottleneck on the amount of training data.

In this work, the proposed ```FGLCD``` makes the following advancement compared with the previous ```Million-AID```:

- manually select and correct the annotation of the samples

- enlarge the size of benchmark: a total of 59994 samples (29998 for training, 29996 for testing)

## Fine-grained Land-Cover Dataset (FGLCD) Overview

- The first dataset for the task of fine-grained land-cover scene classification. Different from conventional remote sensing scene classification datasets, such as ```UCM```, ```AID``` and ```NWPU```, the fine-grained categorization strictly follows the land-use classification standards *[GB/T 21010-2017](https://www.chinesestandard.net/PDF/English.aspx/GBT21010-2017)*. 

- A total of 51 geo-spatial fine-grained categories from 8 coarse-grained categories.

- A total of 59994 samples (29998 for training, 29996 for testing).

## Implementation of Concept Guided Learning (CGL)



## Citation

If you find this work useful for your research, please cite our work as follows:

```BibTeX
@article{bi2025universal,
  title={Universal Fine-grained Visual Categorization by Concept Guided Learning},
  author={Bi, Qi and Yi, Jingjun and Zhan, Haolan and Wei, Ji and Xia, Gui-Song},
  journal={IEEE Transactions on Image Processing},
  volume={34},
  year={2025}
}
```

