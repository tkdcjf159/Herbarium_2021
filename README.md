# [Herbarium 2021 - Half-Earth Challenge - FGVC8 ](https://www.kaggle.com/c/herbarium-2021-fgvc8)

## Description
The Herbarium 2021: Half-Earth Challenge is to identify vascular plant specimens provided by the New York Botanical Garden (NY), Bishop Museum (BPBM), Naturalis Biodiversity Center (NL), Queensland Herbarium (BRI), and Auckland War Memorial Museum (AK).

The Herbarium 2021: Half-Earth Challenge dataset includes more than 2.5M images representing nearly 65,000 species from the Americas and Oceania that have been aligned to a standardized plant list (LCVP v1.0.2).

This dataset has a long tail; there are a minimum of 3 images per species. However, some species can be represented by more than 100 images. This dataset only includes vascular land plants which include lycophytes, ferns, gymnosperms, and flowering plants. The extinct forms of lycophytes are the major component of coal deposits, ferns are indicators of ecosystem health, gymnosperms provide major habitats for animals, and flowering plants provide almost all of our crops, vegetables, and fruits.

The teams with the most accurate models will be contacted with the intention of using them on the unnamed plant collections in the NYBG herbarium and then be assessed by the NYBG plant specialists for accuracy.
![](https://i.postimg.cc/htpxH99f/Herbarium2021.png)

## Background

There are approximately 3,000 herbaria world-wide, and they are massive repositories of plant diversity data. These collections not only represent a vast amount of plant diversity, but since herbarium collections include specimens datingback hundreds of years, they provide snapshots of plant diversity through time. The integrity of the plant is maintained in herbaria as a pressed, dried specimen; a specimen collected nearly two hundred years ago by Darwin looks much thesame as one collected a month ago by an NYBG botanist. All specimens not only maintain their morphological features but also include collection dates and locations, their reproductive state, and the name of the person who collected the specimen. This information, multiplied by millions of plant collections, provides the framework for understanding plant diversity on a massive scale and learning how it has changed over time. The models developed during this competition are an integral first step to speed the pace of species discovery and save the plants of the world.

There are approximately 400,000 known vascular plant species with an estimated 80,000 still to be discovered. Herbaria contain an overwhelming amount of unnamed and new specimens, and with the threats of climate change, we need new toolsto quicken the pace of species discovery. This is more pressing today as a United Nations report indicates that more than one million species are at risk of extinction, and amid this dire prediction is a recent estimate that suggests plants are disappearing more quickly than animals. This year, we have expanded our curated herbarium dataset to vascular plant diversity in the Americas and Oceania.

The most accurate models will be used on the unidentified plant specimens in our herbarium and assessed by our taxonomists thereby producing a tool to quicken the pace of species discovery.

## Data (train + test = 130GB)

The training and test set contain images of herbarium specimens from nearly 65,000 species of vascular plants. Each image contains exactly one specimen. The text labels on the specimen images have been blurred to remove category information in the image.

The data has been approximately split 80%/20% for training/test. Each category has at least 1 instance in both the training and test datasets. Note that the test set distribution is slightly different from the training set distribution. The training set contains species with hundreds of examples, but the test set has the number of examples per species capped at a maximum of 10.

## Approch

###   Base
  - Pytorch
  - model : Resnet34 (use pretrained image from Imagenet)
    -  Due to limited time for experiment, choose Resnet34 as a base model. 
    -  ![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FboNAF7%2FbtqJZnRZi51%2F93XxZMGRtuj1OhSneyuI20%2Fimg.png)
  - optimizer : Adam
  - learning rate : 0.001
  - epochs : 20
  - distributed learning using horovod

###   Base + Data Augmentation([CutMix](https://arxiv.org/abs/1905.04899))
  - Use data augmentation to prevent over-fitting
  - CutMix improves the model robustness against input corruptions and its out-of-distribution detection performances

###   Base + Data Augmentation + Scheduler(Cosine Annealing with warmup)
  - Linear Warmup With Cosine Annealing is a learning rate schedule where we increase the learning rate linearly for updates and then anneal according to a cosine schedule afterwards.
  - examples
     - case1 : CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=500, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=100, gamma=1.0)
     ![](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/src/plot001.png?raw=true)
     - case2 : CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=200, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=50, gamma=0.5)
     ![](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/src/plot002.png?raw=true)
  - hyperparameter
     - warmup_steps = steps_per_epoch * 0.1
     - first_cycle_steps = steps_per_epoch
     - gamma = 0.8
     - max_lr = 0.001
     - min_lr = 1e-6

###   Base + Data Augmentation + Scheduler(Cosine Annealing with warmup) + Data balancing
  - Herbarium 2021 dataset suffers from prominent class im-balance.
  - ![](https://www.kaggleusercontent.com/kf/56556674/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..9oCPYrfWpKAb0FvOySO9jA.OUC5FfKoKthU7ge-W9NBQyUJnmiV8fzaFClZBnHZ1SAqDxc4foyTdZRRQDYCEQUNotI_CqPgApXNpgv-Vf9hIXecZw0i_ZobQfNVQ-oaDDkgzxmuwopY_dTLq7xw5om8VMHqUVLAj8useuvoP4EmBd6Y4eXnlYEA8HGKbV2safiIXuXseCSZI1TsDmXS6Eupqyl7yX3o1j8KkeeSXf5S6Wve-GMgilrLzYWRvfUzmlFGcXfXVKmvFi-3nj11mCB_nAa6XSpSdhqHQn7_Fjcv4RmeTnKxyHBTyW2mfEgBc3EuVaFbtSFr4hnprxeDg4Ih4VRkes9yZuQJKfxWOa93GwOcNbHmONAYtmelMGkv9ldY7UKOnDt3x4qCp40RK23crnLWNKHD6vnH0VbRiE5V8C0jpHaLUo1zJ1T9ao9r1kSE6ePVcFst7TOuLUZ4nwtEgpJD33anSC3C9DiJgAg2mgd4HidwB-S4SkABN0o28oQV3Unxn3sosmv8vc1JLI52KH3wl0ZkKrLPqw5PC_ZpL0H3bUawXMyOkILdth9ixQyfsv7vCJdnXilt36Hlnji9EZEN7mquPw2T83Ky9EGCdvAqQvDTpLEaZ7aheTQeB_v883eAwi4NkmpJBDPMAdSyy4nqY4vUwOtokpHCZXPLjzNlwwfg_KG_sk8szOMcVC4.HYpyBq6rNn8x7LFDBNCwwA/__results___files/__results___5_1.png)
  - apply weight balancing
    
###   Base + Data Augmentation + Scheduler(Cosine Annealing with warmup) + Data balancing + Model chagnge
  - model change : resnet34 to efficientnet-b3 
  - ![](https://1.bp.blogspot.com/-DjZT_TLYZok/XO3BYqpxCJI/AAAAAAAAEKM/BvV53klXaTUuQHCkOXZZGywRMdU9v9T_wCLcBGAs/s1600/image2.png)
  - ![](https://1.bp.blogspot.com/-oNSfIOzO8ko/XO3BtHnUx0I/AAAAAAAAEKk/rJ2tHovGkzsyZnCbwVad-Q3ZBnwQmCFsgCEwYBhgL/s640/image3.png)

## Result
  - Best: Base + Data Augmentation + Scheduler(Cosine Annealing with warmup) + Data balancing + Model chagnge
  - ![](https://github.com/tkdcjf159/Herbarium_2021/blob/main/imgs/train_loss.png)
  - ![](https://github.com/tkdcjf159/Herbarium_2021/blob/main/imgs/val_accuracy.png)
  - kaggle submit result ([F1 score](https://eunsukimme.github.io/ml/2019/10/21/Accuracy-Recall-Precision-F1-score/))
     - ![](https://github.com/tkdcjf159/Herbarium_2021/blob/main/imgs/submit_result.png) 
  
## Future Plans
  - Add [ArcFace Loss](https://arxiv.org/abs/1801.07698) for generating better representation  
