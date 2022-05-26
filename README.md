First of all, thanks to Kaggle and Cornell Lab of Ornithology for organizing this competition.

I was lucky enough to team-up with [UEMU](https://www.kaggle.com/asaliquid1011), it was a fruitful team-up with lots of ideas coming from the both sides. 
We've worked hard until the end and thanks to that we managed to secure 3rd place, congratz to all of the winners!

Considering the situation, it seems necessary to mention that we didn't use any external data except provided 2022 competition dataset, and also we didn't use BirdNet.

Now for our approach, the key points to build strong and reliable pipeline are the following:

* Use SED model & training scheme proposed in [tattaka's 4th place solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243293)
* Use pseudo-labels & hand-labels for small samples class in SED model training
* Use CNN proposed in [2nd place BirdCLEF 2021 solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243463)
* Divide scored birds into two sets and use different loss functions to train models for each one
* Augmentations

Now let's get down to our best finding during competition

### The bird split

When we merged together, we looked on OOFs predictions produced by our models and found out that SED w/ focal-loss performs very differently compared to mentioned CNN trained w/ BCELoss depending on number of training samples

From our observations, SED models w/ focal-loss tend to make more conservative predictions, and due to the loss design they don't miss small classes:

Here in blue you can see SED model w/ focal-loss, and in pink CNN model trained w/ BCE loss

![image](https://user-images.githubusercontent.com/57013219/170329125-532a0640-cb54-4a81-9d8a-fadd4721d6ae.png)

The same can be said about advantage of models which used BCE loss during training over models with focal-loss for the large classes 

![image](https://user-images.githubusercontent.com/57013219/170329065-9a9d4da1-1660-46d4-b25e-9419f451f63d.png)

Therefore, we divided the birds into two groups according to the number of data and manual inspection of distribution plots of target-data as above, and used different models for them.

It appears that it's optimal to include birds with number of training samples >= 10 to the Group1, and all other birds into Group2.

Group1: 14birds  ['jabwar', 'yefcan', 'skylar', 'akiapo', 'apapan', 'barpet', 'elepai', 'iiwi', 'houfin', 'omao', 'warwhe1', 'aniani', 'hawama', 'hawcre'], 
for them we ended up using CNN + SED models which were trained using BCE loss.

Group2: 7birds,  ['crehon', 'ercfra', 'hawgoo', 'hawhaw', 'hawpet1', 'maupar', 'puaioh'], 
for this group we chose to use SED w/ focal-loss.

### CNN model training, Group1 birds (slime part)

For the details of architecture of CNN model, please refer to [2nd place BirdCLEF 2021 solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243463)
 
Since the CNN model was used only for inference on large enough classes, it allowed us to build reliable validation and monitor metrics for Group1 birds only, for these models we used BCE loss to select best models on validation, however with mix-up augmentation model converged on the last epoch, - this fact allowed us to include some models which were trained on full data in the final ensemble.

#### Training strategy

* Use two front-ends:
  * sr: 32000, window_size: 2048, hop_size: 512, fmin: 0, fmax: 16000, mel_bins: 256, power: 2, top_db=None
  * sr: 32000, window_size: 1024, hop_size: 320, fmin: 50, fmax: 14000, mel_bins: 64, power: 2, top_db=None
* Epochs: 40
* backbone: tf_efficnetnet_b0_ns, tf_efficinetnetv2_s_in21k, resnet34, eca_nfnet_l0 
* Optimizer: Adam, lr=3e-4, wd=0
* Scheduler: CosineAnnealing w/o warm-up
* Labels: use union of primary and secondary labels
* Startify data: by primary label
#### Augmentations

The ones that definitely helped
* mix-up (the most impactful one)
* add background noise (same as 2nd place solution 2021)
* spec-augment
* cut-mix (helped, but just a little)

#### Didn't work

* augment only scored birds
* multiply loss for scored bird by 10 
* use weighted BCE w/ weights proportionally to number of class appearance in dataset
* use PCEN
* random power as in [vlomme's 2021 solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243351), pitch-shift
* coord-conv as in [2nd place rainforest solution](https://www.kaggle.com/competitions/rfcx-species-audio-detection/discussion/220760)
* "rating" data didn't introduce much of a difference


### SED model training, Group1&2 birds (UEMU part)
The SED model hasn't changed much from the previous 4th solution.
The main difference is the addition of some augmentations. 

As a result, we think that the score has improved by 0.06 or more in Public LB. 
For the Group1 model and the Group2 model, we changed the loss function, and the other settings were not changed.

We couldn't find a good CV strategy, so most of the settings are decided by watching at public LB.

#### Training strategy 

* Use two front-ends:
  * sr: 32000, window_size: 2048, hop_size: 1024, fmin: 200, fmax: 14000, mel_bins: 224
  * sr: 32000, window_size: 1024, hop_size: 512, fmin: 50, fmax: 14000, mel_bins: 128
* Epochs: 30-40
* Cropsize: 10-15s
* backbone: seresnext26t_32x4d, resnet34, resnest50, tf_efficientnetv2s
* loss_functiuon: BCE2wayloss(Group1), BCEFocal2WayLoss(Group2) as in [kaeruru's public note](https://www.kaggle.com/code/kaerunantoka/birdclef2022-ex005-f0-infer)
* Optimizer: Adam, lr=1e-3, wd=1e-5
* Scheduler: CosineAnnealing w/o warm-up
* Labels: primary label=0.9995, secondary label=0.5000, other=0.0025
* Startify data: by primary label


#### Augmentations

* GaussianSNR
* Nocall Data of trainsoundscape data in the 2021 comp
* Spec-augment
* Random_CUTMIX as in [kaeruru's public note](https://www.kaggle.com/code/kaerunantoka/birdclef2022-ex005-f0-infer)
* random power as in [vlomme's 2021 solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243351)

#### Other
* How to crop data: 
  * Use Pseudo labeling. We decided the time to crop from the probability distribution estimated by pretrained SED model.
* Oversampling for small samples class: 
  * We split the training files by hand and increase data like [5th place solution](https://www.kaggle.com/competitions/birdclef-2022/discussion/327044) for some classes('crehon', 'ercfra', 'hawgoo', 'hawhaw', 'hawpet1', 'maupar', 'puaioh',....)

#### Didn't work
* some augmentations(mixup, Randomlowpassfilter, pitch shift, etc)
* use PCEN
* use weighted BCE w/ weights proportionally to number of class appearance in dataset
* use 'rating' data
* use 'eBird_Taxonomy_v2021' data

### How to choose thresholds: 
  * The default threshold for every of 21 birds was decided by watching LB (in the end we used 0.05 for Group1 birds, but 0.04 gives >0.82 in private LB).
  * We manually set the threshold for "skylar" bird as 0.35, since our models trained with BCE loss predict it reliably.
  * The probability distribution of OOFs prediction for non-targetdata when using focal-loss differs depending on the bird (see below figure),
  so we set the threshold for each bird from Group2 depending on the distribution. We adopted the value of 91 percentile of the distribution for these birds.
  
  ![image](https://user-images.githubusercontent.com/57013219/170404672-c95e0539-21cd-4378-a4cd-75a8c756bdf4.png)


### Final results

For our best public LB submission we used 8 CNN models, 8 SED models for Group1 birds & 12 SED models for Group2 birds (single model here is either model trained on one fold or on the whole data)

| Name                    | Public LB   | Private LB | 
| ----------------------------------------------| ----------------- | ---------- |
| CNN model (no augs, single fold)                    | 0.7715      | 0.7278     |
| CNN model (augs, single fold) | 0.7761 | 0.7359 |
| Best CNN ensemble (CNN only w/ BCE loss) | 0.8327        | 0.7898 |
| SED model (4fold average w/ BCEFocal2wayloss)| 0.8339        | 0.7823 |
| Combine UEMU's SED w/ focal-loss & slime's CNN w/ BCE Loss using bird split mentioned above | 0.8532  | 0.8052    |
| Same as above, but add more CNNs w/ BCE loss and SED w/ BCE loss to group1 birds (best public LB, sub1) | 0.8750  | 0.8126      |
| Safe submission (lower thresholds, sub2) | 0.8556     | 0.8071 |
| Best private LB (add some models which didn't work on public LB) | 0.8707    | 0.8274 |

We also had around 20 subs which score > 0.82 on private LB and > 0.87 on public LB, but we didn't select them since we chose two subs in the following manner, - one is the best public LB and the other one with much lower thresholds to prevent the shake-up, this sub also happend to be placed 3rd, we're happy :)

Ask your questions! :)
