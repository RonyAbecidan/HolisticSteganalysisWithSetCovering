# Official repository for the paper :

## "Using Set Covering to Generate Databases for Holistic Steganalysis"

### released at @[WIFS2022](https://wifs2022.utt.fr/) (Shanghai, China)

![](https://img.shields.io/badge/Official%20-Yes-1E8449.svg) ![](https://img.shields.io/badge/Topic%20-Operational_Steganalysis-2E86C1.svg) ![visitors](https://visitor-badge.glitch.me/badge?page_id=HolisticSteganalysisWithSetCovering)

<p align="center">
  <img src="https://svgshare.com/i/o_E.svg" />
</p>


<br/>

[![](https://img.shields.io/badge/Bibtex-0C0C0C?style=for-the-badge)](#CitingUs)   [![](https://img.shields.io/badge/Article-2E86C1?style=for-the-badge)](https://hal.archives-ouvertes.fr/hal-03840926/)  
### Rony Abecidan, Vincent Itier, Jeremie Boulanger, Patrick Bas, Tomáš Pevný


<br/>

*Abstract : Within an operational framework, covers used by a steganographer are likely to come from different sensors and different processing pipelines than the ones used by researchers for training their steganalysis models. Thus, a performance gap is unavoidable when it comes to out-of-distributions covers, an extremely frequent scenario called Cover Source Mismatch (CSM). Here, we explore a grid of processing pipelines to study the origins of CSM, to better understand it, and to better tackle it. A set-covering greedy algorithm is used to select representative pipelines minimizing the maximum regret between the representative and the pipelines within the set. Our main contribution is a methodology for generating relevant bases able to tackle operational CSM.*

## Comments about the repo : 

- The file ```pipelines.csv``` is a directory of pipeline disclosing their parameters and identifying them precisely with a number.

- The file ```RAW_DATABASE.csv``` contains some information about all the RAW images we used for our experiments. They are all from the database [ALASKA](https://alaska.utt.fr/).

- The file ```FLICKR_BASE.csv``` contains some information about all the FLICKR images we used as our wild base for our last experiment. They are extracted from the website [FLICKR](https://www.flickr.com/) and copyright free.

- The folder ```1-Developing``` contains some code enabling to develop RAW Images like we did.

- The folder ```2-Clustering``` contains some code enabling to extract relevant pipelines from the grid using the greedy set-covering algorithm we used. There is also a playground notebook to help you reproduce some results we obtained in the paper. **Don't hesitate to use our PE/Regret Matrix to derive your own conclusions**.

- To be able to reproduce our experiments and do your own ones, please follow our [Installation Instructions](INSTALL.md)


## Example of covering :

Using a maximum regret radius of 10%, the greedy algorithm returned a set of 5 pipelines. Hence, 5 sources enabling to cover every other source from the grid to an accuracy of 10% in terms of regret. Meaning,

*Whatever the source you consider from the grid, I can always find a representative among the 5 found such that, training on this representative will give me a test performance almost as satisfying as if I trained directly on the original source, the maximum difference of performance being 10%.*

<p align="center">
  <img src="https://svgshare.com/i/oa2.svg" />
</p>

Illustration of the covering obtained with a maximum regret radius of 10%

## Main references

```BibTeX
@article{giboulot:hal-02631559,
  TITLE = {{Effects and Solutions of Cover-Source Mismatch in Image Steganalysis}},
  AUTHOR = {Giboulot, Quentin and Cogranne, R{\'e}mi and Borghys, Dirk and Bas, Patrick},
  URL = {https://hal-utt.archives-ouvertes.fr/hal-02631559},
  JOURNAL = {{Signal Processing: Image Communication}},
  PUBLISHER = {{Elsevier}},
  SERIES = {86},
  YEAR = {2020},
  MONTH = Aug,
  DOI = {10.1016/j.image.2020.115888},
  KEYWORDS = {Steganography ; Steganalysis ; Cover-Source Mismatch ; Image processing ; Image Heterogeneity},
  PDF = {https://hal-utt.archives-ouvertes.fr/hal-02631559/file/ImageCommunication_Final.pdf},
  HAL_ID = {hal-02631559},
  HAL_VERSION = {v1},
}

@inproceedings{giboulot:hal-03694662,
  TITLE = {{The Cover Source Mismatch Problem in Deep-Learning Steganalysis}},
  AUTHOR = {Giboulot, Quentin and Bas, Patrick and Cogranne, R{\'e}mi and Borghys, Dirk},
  URL = {https://hal-utt.archives-ouvertes.fr/hal-03694662},
  BOOKTITLE = {{European Signal Processing Conference}},
  ADDRESS = {Belgrade, Serbia},
  YEAR = {2022},
  MONTH = Aug,
  PDF = {https://hal-utt.archives-ouvertes.fr/hal-03694662/file/Giboulot_EUSIPCO_2022.pdf},
  HAL_ID = {hal-03694662},
  HAL_VERSION = {v1},
}

@inproceedings{cogranne:hal-02147763,
  TITLE = {{The ALASKA Steganalysis Challenge: A First Step Towards Steganalysis ''Into The Wild''}},
  AUTHOR = {Cogranne, R{\'e}mi and Giboulot, Quentin and Bas, Patrick},
  URL = {https://hal.archives-ouvertes.fr/hal-02147763},
  BOOKTITLE = {{ACM IH\&MMSec (Information Hiding \& Multimedia Security)}},
  ADDRESS = {Paris, France},
  SERIES = {ACM IH\&MMSec (Information Hiding \& Multimedia Security)},
  YEAR = {2019},
  MONTH = Jul,
  DOI = {10.1145/3335203.3335726},
  KEYWORDS = {steganography ; steganalysis ; contest ; forensics ; Security and privacy},
  PDF = {https://hal.archives-ouvertes.fr/hal-02147763/file/ALASKA_lesson_learn_Vsubmitted.pdf},
  HAL_ID = {hal-02147763},
  HAL_VERSION = {v1},
}

@inproceedings{sepak,
	title = {Formalizing cover-source mismatch as a robust optimization},
	author = {Šepák, Dominik and Adam, Lukáš and Pevný, Tomáš},
  BOOKTITLE  = {{EUSIPCO: European Signal Processing Conference}},
  MONTH = Sep,
  ADDRESS = {Belgrade, Serbia},
  YEAR = {2022},
}

@inproceedings{10.1145/3437880.3460395,
author = {Butora, Jan and Yousfi, Yassine and Fridrich, Jessica},
title = {How to Pretrain for Steganalysis},
year = {2021},
isbn = {9781450382953},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3437880.3460395},
doi = {10.1145/3437880.3460395},
booktitle = {Proceedings of the 2021 ACM Workshop on Information Hiding and Multimedia Security},
pages = {143–148},
numpages = {6},
keywords = {steganalysis, imagenet, convolutional neural network, JPEG, transfer learning},
location = {Virtual Event, Belgium},
series = {IH&amp;MMSec '21}
}


```

---
## <a name="CitingUs"></a>Citing our paper
### If you wish to refer to our paper,  please use the following BibTeX entry
```BibTeX

@inproceedings{abecidan:hal-03840926,
  TITLE = {{Using Set Covering to Generate Databases for Holistic Steganalysis}},
  AUTHOR = {Abecidan, Rony and Itier, Vincent and Boulanger, J{\'e}r{\'e}mie and Bas, Patrick and Pevn{\'y}, Tom{\'a}{\v s}},
  URL = {https://hal.archives-ouvertes.fr/hal-03840926},
  BOOKTITLE = {{IEEE International Workshop on Information Forensics and Security (WIFS 2022)}},
  ADDRESS = {Shanghai, China},
  YEAR = {2022},
  MONTH = Dec,
  PDF = {https://hal.archives-ouvertes.fr/hal-03840926/file/2022_wifs.pdf},
  HAL_ID = {hal-03840926},
  HAL_VERSION = {v1},
}

```
