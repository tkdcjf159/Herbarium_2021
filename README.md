# [Herbarium 2021 - Half-Earth Challenge - FGVC8 ](https://www.kaggle.com/c/herbarium-2021-fgvc8)

## Description
The Herbarium 2021: Half-Earth Challenge is to identify vascular plant specimens provided by the New York Botanical Garden (NY), Bishop Museum (BPBM), Naturalis Biodiversity Center (NL), Queensland Herbarium (BRI), and Auckland War Memorial Museum (AK).

The Herbarium 2021: Half-Earth Challenge dataset includes more than 2.5M images representing nearly 65,000 species from the Americas and Oceania that have been aligned to a standardized plant list (LCVP v1.0.2).

This dataset has a long tail; there are a minimum of 3 images per species. However, some species can be represented by more than 100 images. This dataset only includes vascular land plants which include lycophytes, ferns, gymnosperms, and flowering plants. The extinct forms of lycophytes are the major component of coal deposits, ferns are indicators of ecosystem health, gymnosperms provide major habitats for animals, and flowering plants provide almost all of our crops, vegetables, and fruits.

The teams with the most accurate models will be contacted with the intention of using them on the unnamed plant collections in the NYBG herbarium and then be assessed by the NYBG plant specialists for accuracy.

![](https://postimg.cc/BjPXFP70)

## Background

There are approximately 3,000 herbaria world-wide, and they are massive repositories of plant diversity data. These collections not only represent a vast amount of plant diversity, but since herbarium collections include specimens datingback hundreds of years, they provide snapshots of plant diversity through time. The integrity of the plant is maintained in herbaria as a pressed, dried specimen; a specimen collected nearly two hundred years ago by Darwin looks much thesame as one collected a month ago by an NYBG botanist. All specimens not only maintain their morphological features but also include collection dates and locations, their reproductive state, and the name of the person who collected the specimen. This information, multiplied by millions of plant collections, provides the framework for understanding plant diversity on a massive scale and learning how it has changed over time. The models developed during this competition are an integral first step to speed the pace of species discovery and save the plants of the world.

There are approximately 400,000 known vascular plant species with an estimated 80,000 still to be discovered. Herbaria contain an overwhelming amount of unnamed and new specimens, and with the threats of climate change, we need new toolsto quicken the pace of species discovery. This is more pressing today as a United Nations report indicates that more than one million species are at risk of extinction, and amid this dire prediction is a recent estimate that suggests plants are disappearing more quickly than animals. This year, we have expanded our curated herbarium dataset to vascular plant diversity in the Americas and Oceania.

The most accurate models will be used on the unidentified plant specimens in our herbarium and assessed by our taxonomists thereby producing a tool to quicken the pace of species discovery.

## Approch

###   Base
  - Pytorch
  - model : Resnet34 (use pretrained image from Imagenet)
    -  Due to limited time for experiment, choose Resnet34 as a base model to train faster in multi-gpu environment
    -  ![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FboNAF7%2FbtqJZnRZi51%2F93XxZMGRtuj1OhSneyuI20%2Fimg.png)
  - optimizer : Adam
  - learning rate : 0.001
  - distributed learning using horovod

###   Base + Data Balancing
  -  balanced sampling (randomly sample same number of images per category per epoch).
  ![](https://www.kaggleusercontent.com/kf/56421146/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..H7qeQNSOj4PDhe44TciSiw.t5MEBtaVDT9bMwVHJoB6P07CSjUWrjFA4nYaxttfU75wHwif0hbRBZ1m3uQ2LIMWYDUwuSubJWF9V9h4RPnO-mz46Gq3yvF5f23hZwrMxWzOis_W8B0X3vMMdPfSTuiFf_z9ScXKgemuUZVPKtHhJ083e-zcriIga3Sq1wzoQ7Vq-M6cvY1NU9RaDtsqyg4beraw-BgEqi6Nf0QRuT9_xW4_-IkbMvF9W7_ObxuL-75LyQxc7wHmQF1Xxk41dQFw-azhMGU-Fc7tiGug-LmAN6whfew8xMLfHmQyLLriTPvS6tyA2_GXXY0GRcSQLA00CBAFSo9huOu9D1U_w_OM20EBjGLxvPpk6JuA-2Yukw4VDg-FM1uhK9LiqSugqa1W-4I22_WFDJRgIkApb1ZoFEiuGXtep-bLH8S_Wd--B4cu7BxGMFktQfiy2O9ql54rJKUkdM3tTY-WMZ16mNo5M-F7ooc2GqG4Dta4sVyehobqu9sQeQGBw7ovd29laEN4l6-_6ExNYepevfoOc1nsca2IXjULzlR4fRYp1yCEZYQpFFqzXQYSx5sE017i-OkiSVLQTwOscoSejmsp6oJqW86amPeaaeXz4rf_E2xeCW2nDT_vk1v_thR1KVmMgRptgUQKTHWS-voMoSwvPrGl0v4-4S4riml-a0CgiPCD9F8.79gWc4YuaByW4sl8IZS5ag/__results___files/__results___5_1.png)

###   Base + Data Balancing + Data Augmentation([CutMix](https://arxiv.org/abs/1905.04899))
  - Use data augmentation to prevent over-fitting
  - CutMix improves the model robustness against input corruptions and its out-of-distribution detection performances

###   Base + Data Balancing + Data Augmentation + Scheduler(Cosine Annealing with warmup)
  - Linear Warmup With Cosine Annealing is a learning rate schedule where we increase the learning rate linearly for updates and then anneal according to a cosine schedule afterwards.
  - examples
     - case1 : CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=500, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=100, gamma=1.0)
     ![](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/src/plot001.png?raw=true)
     - case2 : CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=200, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=50, gamma=0.5)
     ![](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/src/plot002.png?raw=true)
  - hyperparameter
     - warmup_steps = steps_per_epoch * 0.1
     - first_cycle_steps = steps_per_epoch
     - gamma = 0.5
     - max_lr = 0.001
     - min_lr = 1e-6

## Future Plans
  - Add [ArcFace Loss](https://arxiv.org/abs/1801.07698) for generating better representation
  - Change Model Architecture and apply [Self-training with Noisy Student](https://arxiv.org/pdf/1911.04252v4.pdf)
      - Model : EfficientNet
