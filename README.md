<!--
 * @Author: fzy
 * @Date: 2020-03-09 21:53:10
 * @LastEditors: Zhenying
 * @LastEditTime: 2020-12-03 18:58:12
 * @Description: 
 -->
# Awesome-Temporal-Action-Detection-Temporal-Action-Proposal-Generation [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/zhenyingfang/Awesome-Temporal-Action-Detection-Temporal-Action-Proposal-Generation)
Temporal Action Detection &amp; Weakly Supervised Temporal Action Detection &amp; Temporal Action Proposal Generation

-----
**Contents**
<!-- TOC -->
- [Awesome-Temporal-Action-Detection-Temporal-Action-Proposal-Generation](#awesome-temporal-action-detection-temporal-action-proposal-generation)
- [**about pretrained model**](#about-pretrained-model)
- [**ActivityNet Challenge**](#activitynet-challenge)
- [**Papers: Temporal Action Proposal Generation**](#papers-temporal-action-proposal-generation)
  - [2021](#2021) - [2020](#2020) - [2019](#2019) - [2018](#2018) - [2017](#2017) - [before](#before)
- [**Papers: Temporal Action Detection**](#papers-temporal-action-detection)
  - [2021](#2021-1) - [2020](#2020-1) - [2019](#2019-1) - [2018](#2018-1) - [2017](#2017-1) - [before](#before-1)
- [**Papers: Weakly Supervised Temporal Action Detection**](#papers-weakly-supervised-temporal-action-detection)
  - [2021](#2021-2) - [2020](#2020-2) - [2019](#2019-2) - [2018](#2018-2) - [2017](#2017-2)
- [**Papers: Online Action Detection**](#papers-online-action-detection)
  - [2021](#2021-3)


-----
# **about pretrained model**
1. (BSP) [Boundary-sensitive Pre-training for Temporal Localization in Videos](https://arxiv.org/abs/2011.10830) (ICCV 2021)
2. (TSP) [TSP: Temporally-Sensitive Pretraining of Video Encoders for Localization Tasks](https://arxiv.org/abs/2011.11479) (arxiv 2020)

# **ActivityNet Challenge and talks**
1. (2021) [AcitvityNet 2021](http://activity-net.org/challenges/2021/challenge.html)
2. (2021) [Transformer在时序行为检测中的应用 & 基于自监督学习的半监督时序行为检测](https://www.techbeat.net/talk-info?id=545) (DAMO Academy, Alibaba Group)

# **Papers: Temporal Action Proposal Generation**

## 2021
1. (BSN++) [BSN++: Complementary Boundary Regressor with Scale-Balanced RelationModeling for Temporal Action Proposal Generation](https://arxiv.org/abs/2009.07641) (AAAI 2021) [Author's Zhihu](https://zhuanlan.zhihu.com/p/344065976)
2. (RTD-Net) [Relaxed Transformer Decoders for Direct Action Proposal Generation](https://arxiv.org/abs/2102.01894) (ICCV 2021) [code](https://github.com/MCG-NJU/RTD-Action) [Zhihu](https://zhuanlan.zhihu.com/p/363133304)
3. (TCANet) [Temporal Context Aggregation Network for Temporal Action Proposal Refinement](https://arxiv.org/abs/2103.13141) (CVPR 2021) [Zhihu](https://zhuanlan.zhihu.com/p/358754602)
4. [Augmented Transformer with Adaptive Graph for Temporal Action Proposal Generation](https://arxiv.org/abs/2103.16024) (arxiv 2021)
5. [Self-Supervised Learning for Semi-Supervised Temporal Action Proposal](https://arxiv.org/abs/2104.03214) (CVPR 2021) [code](https://github.com/wangxiang1230/SSTAP)
6. (TAPG) [Temporal Action Proposal Generation with Transformers](https://arxiv.org/abs/2105.12043) (arxiv 2021)
7. (AEN) [Agent-Environment Network for Temporal Action Proposal Generation](https://arxiv.org/abs/2107.08323) (ICASSP 2021)

## 2020

1. **VALSE talk by Tianwei Lin** (2020.03.18) [link](https://pan.baidu.com/s/18uPJX3l69qJHaYOdeJ0IQw?errmsg=Auth+Login+Sucess&errno=0&ssnerror=0&) (7y8g)
2. (RapNet) **Accurate Temporal Action Proposal Generation with Relation-Aware Pyramid Network** (AAAI 2020) [pre-paper 2019 ActivityNet task-1 2nd](https://arxiv.org/abs/1908.03448)
3. (DBG) **Fast Learning of Temporal Action Proposal via Dense Boundary Generator** (AAAI 2020) [paper](https://arxiv.org/abs/1911.04127) [code.TensorFlow](https://github.com/TencentYoutuResearch/ActionDetection-DBG)
4. (BC-GNN) **Boundary Content Graph Neural Network for Temporal Action Proposal Generation** (ECCV 2020) [paper](https://arxiv.org/abs/2008.01432v1)
5. [Bottom-Up Temporal Action Localization with Mutual Regularization](https://arxiv.org/abs/2002.07358) (ECCV 2020) [code.TensorFlow](https://github.com/PeisenZhao/Bottom-Up-TAL-with-MR)
6. (TSI) [TSI: Temporal Scale Invariant Network for Action Proposal Generation](https://openaccess.thecvf.com/content/ACCV2020/html/Liu_TSI_Temporal_Scale_Invariant_Network_for_Action_Proposal_Generation_ACCV_2020_paper.html) (ACCV 2020)

## 2019

1. (SRG) **SRG: Snippet Relatedness-based Temporal Action Proposal Generator** (IEEE Trans 2019) [paper](https://arxiv.org/abs/1911.11306)
2. (DPP) **Deep Point-wise Prediction for Action Temporal Proposal** (ICONIP 2019) [paper](https://arxiv.org/abs/1909.07725) [code.PyTorch](https://github.com/liluxuan1997/DPP)
3. (semi-supervised) **Learning Temporal Action Proposals With Fewer Labels** (ICCV 2019) [paper](https://arxiv.org/abs/1910.01286)
4. (BMN) **BMN: Boundary-Matching Network for Temporal Action Proposal Generation** (ICCV 2019) [paper](https://arxiv.org/abs/1907.09702) [code.PaddlePaddle](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/video) [code.PyTorch_unofficial](https://github.com/JJBOY/BMN-Boundary-Matching-Network)
5. (MGG) **Multi-granularity Generator for Temporal Action Proposal** (CVPR 2019) [paper](https://arxiv.org/abs/1811.11524)
6. **Investigation on Combining 3D Convolution of Image Data and Optical Flow to Generate Temporal Action Proposals** (2019 CVPR Workshop) [paper](https://arxiv.org/abs/1903.04176)
7. (CMSN) **CMSN: Continuous Multi-stage Network and Variable Margin Cosine Loss for Temporal Action Proposal Generation** (arxiv 2019) [paper](https://arxiv.org/abs/1911.06080)
8. **A high performance computing method for accelerating temporal action proposal generation** (arxiv 2019) [paper](https://arxiv.org/abs/1906.06496)
9. **Multi-Granularity Fusion Network for Proposal and Activity Localization: Submission to ActivityNet Challenge 2019 Task 1 and Task 2** (ActvityNet challenge 2019) [paper](https://arxiv.org/abs/1907.12223)
10. [Joint Learning of Local and Global Context for Temporal Action Proposal Generation](https://ieeexplore.ieee.org/abstract/document/8941024) (TCSVT 2019)

## 2018

1. (CTAP) **CTAP: Complementary Temporal Action Proposal Generation** (ECCV 2018) [paper](https://arxiv.org/abs/1807.04821) [code.TensorFlow](https://github.com/jiyanggao/CTAP)
2. (BSN) **BSN: Boundary Sensitive Network for Temporal Action Proposal Generation** (ECCV 2018) [paper](https://arxiv.org/abs/1806.02964) [code.TensorFlow](https://github.com/wzmsltw/BSN-boundary-sensitive-network) [code.PyTorch](https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch)
3. (SAP) **SAP: Self-Adaptive Proposal Model for Temporal Action Detection based on Reinforcement Learning** (AAAI 2018) [paper](https://github.com/hjjpku/Action_Detection_DQN/blob/master/camera%20ready.pdf) [code.Torch](https://github.com/hjjpku/Action_Detection_DQN)

## 2017

1. (TURN TAP) **TURN TAP: Temporal Unit Regression Network for Temporal Action Proposals** (ICCV 2017) [paper](https://arxiv.org/abs/1703.06189) [code.TensorFlow](https://github.com/jiyanggao/TURN-TAP)
2. (SST) **SST: Single-Stream Temporal Action Proposals** (CVPR 2017) [paper](http://vision.stanford.edu/pdf/buch2017cvpr.pdf) [code.theano](https://github.com/shyamal-b/sst/) [code.TensorFlow](https://github.com/JaywongWang/SST-Tensorflow)
3. **YoTube: Searching Action Proposal via Recurrent and Static Regression Networks** (IEEE Trans 2017) [paper](https://arxiv.org/abs/1706.08218)
4. **A Pursuit of Temporal Accuracy in General Activity Detection** (arxiv 2017) [paper](https://arxiv.org/abs/1703.02716) [code.PyTorch](https://github.com/yjxiong/action-detection)

## before

1. (DAPs) **DAPs: Deep Action Proposals for Action Understanding** (ECCV 2016) [paper](https://drive.google.com/file/d/0B0ZXjo_p8lHBcjh1WDlmYVN3R2M/view) [code](https://github.com/escorciav/deep-action-proposals)

----
# **Papers: Temporal Action Detection**

## 2021
1. (activity graph transformer) [Activity Graph Transformer for Temporal Action Localization](https://arxiv.org/abs/2101.08540) (arxiv 2021) [project](https://www.sfu.ca/~mnawhal/projects/agt.html) [code](https://github.com/Nmegha2601/activitygraph_transformer)
2. [Coarse-Fine Networks for Temporal Activity Detection in Videos](https://arxiv.org/abs/2103.01302) (CVPR 2021) [code](https://github.com/kkahatapitiya/Coarse-Fine-Networks)
3. (MLAD) [Modeling Multi-Label Action Dependencies for Temporal Action Localization](https://arxiv.org/abs/2103.03027) (CVPR 2021)
4. (PcmNet) [PcmNet: Position-Sensitive Context Modeling Network for Temporal Action Localization](https://arxiv.org/abs/2103.05270) (Tip 2021)
5. (AFSD) [Learning Salient Boundary Feature for Anchor-free Temporal Action Localization](https://arxiv.org/abs/2103.13137) (CVPR 2021) [code](https://github.com/TencentYoutuResearch/ActionDetection-AFSD?utm_source=catalyzex.com)
6. [Low-Fidelity End-to-End Video Encoder Pre-training for Temporal Action Localization](https://arxiv.org/abs/2103.15233) (arxiv 2021)
7. [Read and Attend: Temporal Localisation in Sign Language Videos](https://arxiv.org/abs/2103.16481) (CVPR 2021) (Sign Language Videos)
8. [Low Pass Filter for Anti-aliasing in Temporal Action Localization](https://arxiv.org/abs/2104.11403) (arxiv 2021)
9. [FineAction: A Fined Video Dataset for Temporal Action Localization](https://arxiv.org/abs/2105.11107) (One track of DeeperAction Workshop@ICCV2021) [Homepage](https://deeperaction.github.io/fineaction/)
10. (TadTR) [End-to-end Temporal Action Detection with Transformer](https://arxiv.org/abs/2106.10271) (arxiv 2021) [code](https://github.com/xlliu7/TadTR)
11. [Three Birds with One Stone: Multi-Task Temporal Action Detection via Recycling Temporal Annotations](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Three_Birds_with_One_Stone_Multi-Task_Temporal_Action_Detection_via_CVPR_2021_paper.html) (CVPR 2021)
12. [Proposal Relation Network for Temporal Action Detection](https://arxiv.org/abs/2106.11812) (CVPRW 2021)
13. [Exploring Stronger Feature for Temporal Action Localization](https://arxiv.org/abs/2106.13014) (CVPRW 2021)
14. (SRF-Net) [SRF-Net: Selective Receptive Field Network for Anchor-Free Temporal Action Detection](https://arxiv.org/abs/2106.15258) (ICASSP 2021)
15. [RGB Stream Is Enough for Temporal Action Detection](https://arxiv.org/abs/2107.04362) (arxiv 2021)
16. (AVFusion) [Hear Me Out: Fusional Approaches for Audio Augmented Temporal Action Localization](https://arxiv.org/pdf/2106.14118v1.pdf) (arxiv 2021) [Code](https://github.com/skelemoa/tal-hmo)
17. [Transferable Knowledge-Based Multi-Granularity Aggregation Network for Temporal Action Localization: Submission to ActivityNet Challenge 2021](https://arxiv.org/abs/2107.12618) (HACS challenge 2021)
18. [Enriching Local and Global Contexts for Temporal Action Localization](https://arxiv.org/abs/2107.12960) (ICCV 2021)
19. [Class Semantics-based Attention for Action Detection](https://arxiv.org/abs/2109.02613) (ICCV 2021)

## 2020

1. (G-TAD) **G-TAD: Sub-Graph Localization for Temporal Action Detection** (CVPR 2020) [paper](https://arxiv.org/abs/1911.11462) [code.PyTorch](https://github.com/frostinassiky/gtad) [video](https://www.youtube.com/watch?v=BlPxnDcykUo)
2. (AGCN-P-3DCNNs) **Graph Attention based Proposal 3D ConvNets for Action Detection** (AAAI 2020) [paper](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-LiJ.1424.pdf)
3. (PBRNet) **Progressive Boundary Refinement Network for Temporal Action Detection** (AAAI 2020) [paper](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-LiuQ.4870.pdf)
4. (TsaNet) **Scale Matters: Temporal Scale Aggregation Network for Precise Action Localization in Untrimmed Videos** (ICME 2020) [paper](https://arxiv.org/abs/1908.00707)
5. **Constraining Temporal Relationship for Action Localization** (arxiv 2020) [paper](https://arxiv.org/abs/2002.07358)
6. (CBR-Net) **CBR-Net: Cascade Boundary Refinement Network for Action Detection: Submission to ActivityNet Challenge 2020 (Task 1)** (ActivityNet Challenge 2020) [paper](https://arxiv.org/abs/2006.07526v2)
7. [Temporal Action Localization with Variance-Aware Networks](https://arxiv.org/abs/2008.11254) (arxiv 2020)
8. [Boundary Uncertainty in a Single-Stage Temporal Action Localization Network](https://arxiv.org/abs/2008.11170) (arxiv 2020, Tech report)
9. [Revisiting Anchor Mechanisms for Temporal Action Localization](https://arxiv.org/abs/2008.09837) (Tip 2020) [code.PyTorch](https://github.com/VividLe/A2Net?utm_source=catalyzex.com)
10. (C-TCN) [Deep Concept-wise Temporal Convolutional Networks for Action Localization](https://arxiv.org/abs/1908.09442) (ACM MM 2020) [code.PaddlePaddle](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/video)
11. (MLTPN) [Multi-Level Temporal Pyramid Network for Action Detection](https://arxiv.org/abs/2008.03270) (PRCV 2020)
12. (SALAD) [SALAD: Self-Assessment Learning for Action Detection](https://arxiv.org/abs/2011.06958) (arxiv 2020)
13. [Multi-shot Temporal Event Localization: a Benchmark](https://arxiv.org/abs/2012.09434) (arxiv 2020) [project](https://songbai.site/muses/) [code](https://github.com/xlliu7/MUSES) [dataset](https://songbai.site/muses/)

## 2019

1. (CMS-RC3D) **Contextual Multi-Scale Region Convolutional 3D Network for Activity Detection** (ICCVBIC 2019) [paper](https://arxiv.org/abs/1801.09184)
2. (TGM) **Temporal Gaussian Mixture Layer for Videos** (ICML 2019) [paper](https://arxiv.org/abs/1803.06316) [code.PyTorch](https://github.com/piergiaj/tgm-icml19)
3. (Decouple-SSAD) **Decoupling Localization and Classification in Single Shot Temporal Action Detection** (ICME 2019) [paper](https://arxiv.org/abs/1904.07442) [code.TensorFlow](https://github.com/HYPJUDY/Decouple-SSAD)
4. **Exploring Feature Representation and Training strategies in Temporal Action Localization** (ICIP 2019) [paper](https://arxiv.org/abs/1905.10608)
5. (PGCN) **Graph Convolutional Networks for Temporal Action Localization** (ICCV 2019) [paper](https://arxiv.org/abs/1909.03252) [code.PyTorch](https://github.com/Alvin-Zeng/PGCN)
6. (S-2D-TAN) **Learning Sparse 2D Temporal Adjacent Networks for Temporal Action Localization** (ICCV 2019) (*winner solution for the HACS Temporal Action Localization Challenge at ICCV 2019*) [paper](https://arxiv.org/abs/1912.03612) 
   - (2D-TAN) **Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language** (AAAI 2020) [paper](https://arxiv.org/abs/1912.03590) [code.PyTorch](https://github.com/microsoft/2D-TAN)
7. (LCDC) **Learning Motion in Feature Space: Locally-Consistent Deformable Convolution Networks for Fine-Grained Action Detection** (ICCV 2019) [paper](https://arxiv.org/abs/1811.08815) [slide](https://knmac.github.io/projects/lcdc/LCDC_slides_extended.pdf) [code.TensorFlow](https://github.com/knmac/LCDC_release)
8. (BLP) **BLP -- Boundary Likelihood Pinpointing Networks for Accurate Temporal Action Localization** (ICASSP 2019) [paper](https://arxiv.org/abs/1811.02189)
9. (GTAN) **Gaussian Temporal Awareness Networks for Action Localization** (CVPR 2019) [paper](https://arxiv.org/abs/1909.03877)
10. **Temporal Action Localization using Long Short-Term Dependency** (arxiv 2019) [paper](https://arxiv.org/abs/1911.01060)
11. **Relation Attention for Temporal Action Localization** (IEEE Trans TMM 2019) [paper](https://ieeexplore.ieee.org/document/8933113/versions)
12. (AFO-TAD) **AFO-TAD: Anchor-free One-Stage Detector for Temporal Action Detection** (arxiv 2019) [paper](https://arxiv.org/abs/1910.08250)
13. (DBS) **Video Imprint Segmentation for Temporal Action Detection in Untrimmed Videos** (AAAI 2019) [paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4846)

## 2018

1. **Diagnosing Error in Temporal Action Detectors** (ECCV 2018) [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Humam_Alwassel_Diagnosing_Error_in_ECCV_2018_paper.pdf)
2. (ETP) **Precise Temporal Action Localization by Evolving Temporal Proposals** (ICMR 2018) [paper](https://arxiv.org/abs/1804.04803)
3. (Action Search) **Action Search: Spotting Actions in Videos and Its Application to Temporal Action Localization** (ECCV 2018) [paper](https://arxiv.org/abs/1706.04269) [code.TensorFlow](https://github.com/HumamAlwassel/action-search)
4. (TAL-Net) **Rethinking the Faster R-CNN Architecture for Temporal Action Localization** (CVPR 2018) [paper](https://arxiv.org/abs/1804.07667)
5. **One-shot Action Localization by Learning Sequence Matching Network** (CVPR 2018) [paper](http://www.porikli.com/mysite/pdfs/porikli%202018%20-%20One-shot%20action%20localization%20by%20learning%20sequence%20matching%20network.pdf)
6. **Temporal Action Detection by Joint Identification-Verification** (arxiv 2018) [paper](https://arxiv.org/abs/1810.08375)
7. (TPC) **Exploring Temporal Preservation Networks for Precise Temporal Action Localization** (AAAI 2018) [paper](https://arxiv.org/abs/1708.03280)
8. (SAP) **A Self-Adaptive Proposal Model for Temporal Action Detection based on Reinforcement Learning** (AAAI 2018) [paper](https://arxiv.org/abs/1706.07251) [code.Torch](https://github.com/hjjpku/Action_Detection_DQN)

## 2017

1. (TCN) **Temporal Context Network for Activity Localization in Videos** (ICCV 2017) [paper](https://arxiv.org/abs/1708.02349) [code.caffe](https://github.com/vdavid70619/TCN)
2. (SSN) **Temporal Action Detection with Structured Segment Networks** (ICCV 2017) [paper](https://arxiv.org/abs/1704.06228) [code.PyTorch](https://github.com/yjxiong/action-detection)
3. (R-C3D) **R-C3D: Region Convolutional 3D Network for Temporal Activity Detection** (ICCV 2017) [paper](https://arxiv.org/abs/1703.07814) [code.caffe](https://github.com/VisionLearningGroup/R-C3D) [code.PyTorch](https://github.com/sunnyxiaohu/R-C3D.pytorch)
4. (TCNs) **Temporal Convolutional Networks for Action Segmentation and Detection** (CVPR 2017) [paper](https://arxiv.org/abs/1611.05267) [code.TensorFlow](https://github.com/colincsl/TemporalConvolutionalNetworks)
5. (SMS) **Temporal Action Localization by Structured Maximal Sums** (CVPR 2017) [paper](https://arxiv.org/abs/1704.04671) [code](https://github.com/shallowyuan/struct-max-sum)
6. (SCC) **SCC: Semantic Context Cascade for Efficient Action Detection** (CVPR 2017) [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Heilbron_SCC_Semantic_Context_CVPR_2017_paper.pdf)
7. (CDC) **CDC: Convolutional-De-Convolutional Networks for Precise Temporal Action Localization in Untrimmed Videos** (CVPR 2017) [paper](https://arxiv.org/abs/1703.01515) [code](https://bitbucket.org/columbiadvmm/cdc/src/master/) [project](http://www.ee.columbia.edu/ln/dvmm/researchProjects/cdc/cdc.html)
8. (SS-TAD) **End-to-End, Single-Stream Temporal ActionDetection in Untrimmed Videos** (BMVC 2017) [paper](http://vision.stanford.edu/pdf/buch2017bmvc.pdf) [code.PyTorch](https://github.com/shyamal-b/ss-tad/)
9. (CBR) **Cascaded Boundary Regression for Temporal Action Detection** (BMVC 2017) [paper](https://arxiv.org/abs/1705.01180) [code.TensorFlow](https://github.com/jiyanggao/CBR)
10. (SSAD) **Single Shot Temporal Action Detection** (ACM MM 2017) [paper](https://arxiv.org/abs/1710.06236)

## before

1. (PSDF) **Temporal Action Localization with Pyramid of Score Distribution Features** (CVPR 2016) [paper](https://www.zpascal.net/cvpr2016/Yuan_Temporal_Action_Localization_CVPR_2016_paper.pdf)
2. **Temporal Action Detection using a Statistical Language Model** (CVPR 2016) [paper](https://www.zpascal.net/cvpr2016/Richard_Temporal_Action_Detection_CVPR_2016_paper.pdf) [code](https://github.com/alexanderrichard/squirrel)
3. (S-CNN) **Temporal Action Localization in Untrimmed Videos via Multi-stage CNNs** (CVPR 2016) [paper](https://arxiv.org/abs/1601.02129) [code](https://github.com/zhengshou/scnn/) [project](http://www.ee.columbia.edu/ln/dvmm/researchProjects/cdc/scnn.html)
4. **End-to-end Learning of Action Detection from Frame Glimpses in Videos** (CVPR 2016) [paper](https://arxiv.org/abs/1511.06984) [code](https://github.com/syyeung/frameglimpses)

----
# **Papers: Weakly Supervised Temporal Action Detection**

## 2021
1. [A Hybrid Attention Mechanism for Weakly-Supervised Temporal Action Localization](https://arxiv.org/abs/2101.00545). (AAAI 2021)
2. [Cross-Attentional Audio-Visual Fusion for Weakly-Supervised Action Localization](https://openreview.net/forum?id=hWr3e3r-oH5) (ICLR 2021)
3. [Weakly-supervised Temporal Action Localization by Uncertainty Modeling](https://arxiv.org/abs/2006.07006) (AAAI 2021) [code](https://github.com/Pilhyeon/WTAL-Uncertainty-Modeling)
4. (TS-PCA) [The Blessings of Unlabeled Background in Untrimmed Videos](https://arxiv.org/abs/2103.13183) (CVPR 2021) [code](https://github.com/aliyun/The-Blessings-of-Unlabeled-Background-in-Untrimmed-Videos)
5. (ACSNet) [ACSNet: Action-Context Separation Network for Weakly Supervised Temporal Action Localization](https://arxiv.org/abs/2103.15088) (AAAI 2021)
6. (CoLA) [CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning](https://arxiv.org/abs/2103.16392) (CVPR 2021)
7. [Weakly Supervised Temporal Action Localization Through Learning Explicit Subspaces for Action and Context](https://arxiv.org/abs/2103.16155) (AAAI 2021)
8. [Adaptive Mutual Supervision for Weakly-Supervised Temporal Action Localization](https://arxiv.org/abs/2104.02357) (arxiv 2021)
9. [ACM-Net: Action Context Modeling Network for Weakly-Supervised Temporal Action Localization](https://arxiv.org/abs/2104.02967) (arxiv 2021, submitted to Tip) [code](https://github.com/ispc-lab/ACM-Net)
10. (AUMN) [Action Unit Memory Network for Weakly Supervised Temporal Action Localization](https://arxiv.org/abs/2104.14135) (CVPR 2021)
11. (ASL) [Weakly Supervised Action Selection Learning in Video](https://arxiv.org/abs/2105.02439) (CVPR 2021)
12. (ActShufNet) [Action Shuffling for Weakly Supervised Temporal Localization](https://arxiv.org/abs/2105.04208) (arxiv 2021)
13. [Few-Shot Action Localization without Knowing Boundaries](https://arxiv.org/abs/2106.04150) (arxiv 2021)
14. [Uncertainty Guided Collaborative Training for Weakly Supervised Temporal Action Detection](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_Uncertainty_Guided_Collaborative_Training_for_Weakly_Supervised_Temporal_Action_Detection_CVPR_2021_paper.html) (CVPR 2021)
15. [Two-Stream Consensus Network: Submission to HACS Challenge 2021Weakly-Supervised Learning Track](https://arxiv.org/abs/2106.10829) (CVPRW 2021)
16. [Weakly-Supervised Temporal Action Localization Through Local-Global Background Modeling](https://arxiv.org/abs/2106.11811) (CVPRW 2021)
17. [Cross-modal Consensus Network for Weakly Supervised Temporal Action Localization](https://arxiv.org/abs/2107.12589) (ACM MM 2021)
18. [Learning Action Completeness from Points for Weakly-supervised Temporal Action Localization](https://arxiv.org/abs/2108.05029) (ICCV 2021) [code](https://github.com/Pilhyeon/Learning-Action-Completeness-from-Points)
19. [Deep Motion Prior for Weakly-Supervised Temporal Action Localization](https://arxiv.org/abs/2108.05607) (submit to Tip 2021) [project](https://sites.google.com/view/mengcao/publication/dmp-net?authuser=0)

## 2020

1. (WSGN) **Weakly Supervised Gaussian Networks for Action Detection** (WACV 2020) [paper](https://arxiv.org/abs/1904.07774)
2. **Weakly Supervised Temporal Action Localization Using Deep Metric Learning** (WACV 2020) [paper](https://arxiv.org/abs/2001.07793)
3. **Action Graphs: Weakly-supervised Action Localization with Graph Convolution Networks** (WACV 2020) [paper](https://arxiv.org/abs/2002.01449)
4. (DGAM) **Weakly-Supervised Action Localization by Generative Attention Modeling** (CVPR 2020) [paper](https://arxiv.org/abs/2003.12424) [code.PyTorch](https://github.com/bfshi/DGAM-Weakly-Supervised-Action-Localization)
5. (EM-MIL) **Weakly-Supervised Action Localization with Expectation-Maximization Multi-Instance Learning** (ECCV 2020) [paper](https://arxiv.org/abs/2004.00163)
6. **Relational Prototypical Network for Weakly Supervised Temporal ActionLocalization** (AAAI 2020) [paper](https://aaai.org/Papers/AAAI/2020GB/AAAI-HuangL.1235.pdf)
7. (BaS-Net) **Background Suppression Networkfor Weakly-supervised Temporal Action Localization** (AAAI 2020) [paper](https://arxiv.org/abs/1911.09963) [code.PyTorch](https://github.com/Pilhyeon/BaSNet-pytorch)
8. **Background Modeling via Uncertainty Estimation for Weakly-supervised Action Localization** (arxiv 2020) [paper](https://arxiv.org/abs/2006.07006) [code.PyTorch](https://github.com/Pilhyeon/Background-Modeling-via-Uncertainty-Estimation)
9. (A2CL-PT) **Adversarial Background-Aware Loss for Weakly-supervised Temporal Activity Localization** (ECCV 2020) [paper](https://arxiv.org/abs/2007.06643) [code.PyTorch](https://github.com/MichiganCOG/A2CL-PT)
10. **Weakly Supervised Temporal Action Localization with Segment-Level Labels** (arxiv 2020)
11. (ECM) **Equivalent Classification Mapping for Weakly Supervised Temporal Action Localization** (arxiv 2020) [paper](https://arxiv.org/abs/2008.07728v1)
12. [Two-Stream Consensus Network for Weakly-Supervised Temporal Action Localization](https://arxiv.org/abs/2010.11594v1) (ECCV 2020 spotlight)
13. [Learning Temporal Co-Attention Models for Unsupervised Video Action Localization](https://openaccess.thecvf.com/content_CVPR_2020/html/Gong_Learning_Temporal_Co-Attention_Models_for_Unsupervised_Video_Action_Localization_CVPR_2020_paper.html) (CVPR 2020)
14. [Action Completeness Modeling with Background Aware Networks for Weakly-Supervised Temporal Action Localization](https://dl.acm.org/doi/abs/10.1145/3394171.3413687) (ACM MM 2020)
15. (D2-Net) [D2-Net: Weakly-Supervised Action Localization via Discriminative Embeddingsand Denoised Activations](https://arxiv.org/abs/2012.06440) (arxiv 2020) (THUMOS'14 mAP@0.5: 35.9)
16. (SF-Net) [SF-Net: Single-Frame Supervision for Temporal Action Localization](https://arxiv.org/abs/2003.06845) (ECCV 2020) [code.PyTorch](https://github.com/Flowerfan/SF-Net)
17. [Point-Level Temporal Action Localization: Bridging Fully-supervised Proposals to Weakly-supervised Losses](https://arxiv.org/abs/2012.08236) (arxiv 2020)
18. [Transferable Knowledge-Based Multi-Granularity Fusion Network for Weakly Supervised Temporal Action Detection](https://ieeexplore.ieee.org/abstract/document/9105103/keywords#keywords) (TMM 2020)
19. [ActionBytes: Learning From Trimmed Videos to Localize Actions](https://openaccess.thecvf.com/content_CVPR_2020/html/Jain_ActionBytes_Learning_From_Trimmed_Videos_to_Localize_Actions_CVPR_2020_paper.html) (CVPR 2020)

## 2019

1. (AdapNet) **AdapNet: Adaptability Decomposing Encoder-Decoder Network for Weakly Supervised Action Recognition and Localization** (IEEE Transactions on Neural Networks and Learning Systems) [paper](https://arxiv.org/abs/1911.11961)
2. **Breaking Winner-Takes-All: Iterative-Winners-Out Networks for Weakly Supervised Temporal Action Localization** (IEEE Transactions on Image Processing) [paper](https://tanmingkui.github.io/files/publications/Breaking.pdf)
3. **Weakly-Supervised Temporal Localization via Occurrence Count Learning** (ICML 2019) [paper](https://arxiv.org/abs/1905.07293) [code.TensorFlow](https://github.com/SchroeterJulien/ICML-2019-Weakly-Supervised-Temporal-Localization-via-Occurrence-Count-Learning)
4. (MAAN) **Marginalized Average Attentional Network for Weakly-Supervised Learning** (ICLR 2019) [paper](https://arxiv.org/abs/1905.08586) [code.PyTorch](https://github.com/yyuanad/MAAN)
5. **Weakly-supervised Action Localization with Background Modeling** (ICCV 2019) [paper](https://arxiv.org/abs/1908.06552)
6. (TSM) **Temporal Structure Mining for Weakly Supervised Action Detection** (ICCV 2019) [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Temporal_Structure_Mining_for_Weakly_Supervised_Action_Detection_ICCV_2019_paper.pdf)
7. (CleanNet) **Weakly Supervised Temporal Action Localization through Contrast basedEvaluation Networks** (ICCV 2019) [paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Weakly_Supervised_Temporal_Action_Localization_Through_Contrast_Based_Evaluation_Networks_ICCV_2019_paper.html)
8. (3C-Net) **3C-Net: Category Count and Center Loss for Weakly-Supervised Action Localization** (ICCV 2019) [paper](https://arxiv.org/abs/1908.08216) [code.PyTorch](https://github.com/naraysa/3c-net)
9. (CMCS) **Completeness Modeling and Context Separation for Weakly SupervisedTemporal Action Localization** (CVPR 2019) [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Completeness_Modeling_and_Context_Separation_for_Weakly_Supervised_Temporal_Action_CVPR_2019_paper.pdf) [code.PyTorch](https://github.com/Finspire13/CMCS-Temporal-Action-Localization)
10. (RefineLoc) **RefineLoc: Iterative Refinement for Weakly-Supervised Action Localization** (arxiv 2019) [paper](https://arxiv.org/abs/1904.00227) [homepage](http://humamalwassel.com/publication/refineloc/)
11. (TTC-Loc) **Towards Train-Test Consistency for Semi-supervised Temporal Action Localization** (arxiv 2019) (version v3 for LPAT) [paper](https://arxiv.org/abs/1910.11285v3)
12. (ASSG) **Adversarial Seeded Sequence Growing for Weakly-Supervised Temporal Action Localization** (ACM MM 2019) [paper](https://arxiv.org/abs/1908.02422)
13. (TSRNet) **Learning Transferable Self-attentive Representations for Action Recognition in Untrimmed Videos with Weak Supervision** (AAAI 2019) [paper](https://arxiv.org/abs/1902.07370)
14. (STAR) **Segregated Temporal Assembly Recurrent Networks for Weakly Supervised Multiple Action Detection** (AAAI 2019) [paper](https://arxiv.org/abs/1811.07460)

## 2018

1. [Weakly Supervised Temporal Action Detection with Shot-Based Temporal Pooling Network](https://link.springer.com/chapter/10.1007/978-3-030-04212-7_37) (ICONIP 2018)
2. (W-TALC) [W-TALC: Weakly-supervised Temporal Activity Localization and Classification](https://arxiv.org/abs/1807.10418) (ECCV 2018) [code.PyTorch](https://github.com/sujoyp/wtalc-pytorch?utm_source=catalyzex.com)
3. (AutoLoc) [AutoLoc: Weakly-supervised Temporal Action Localization](https://arxiv.org/abs/1807.08333) (ECCV 2018) [code](https://github.com/zhengshou/AutoLoc?utm_source=catalyzex.com)
4. (STPN) [Weakly Supervised Action Localization by Sparse Temporal Pooling Network](https://arxiv.org/abs/1712.05080) (CVPR 2018) [code](https://github.com/demianzhang/weakly-action-localization?utm_source=catalyzex.com)
5. [Step-by-step Erasion, One-by-one Collection: A Weakly Supervised Temporal Action Detector](https://arxiv.org/abs/1807.02929) (ACM MM 2018)
6. (CPMN) [Cascaded Pyramid Mining Network for Weakly Supervised Temporal Action Localization](https://arxiv.org/abs/1810.11794) (accv 2018)

## 2017

1. (Hide-and-Seek) [Hide-and-Seek: Forcing a Network to be Meticulous for
Weakly-supervised Object and Action Localization](https://arxiv.org/abs/1704.04232) (ICCV 2017)
2. (UntrimmedNets) [UntrimmedNets for Weakly Supervised Action Recognition and Detection](https://arxiv.org/abs/1703.03329) (CVPR 2017) [code](https://github.com/wanglimin/UntrimmedNet)

----
# **Papers: Online Action Detection**

## 2021

1. (WOAD) [WOAD: Weakly Supervised Online Action Detection in Untrimmed Videos](https://openaccess.thecvf.com/content/CVPR2021/html/Gao_WOAD_Weakly_Supervised_Online_Action_Detection_in_Untrimmed_Videos_CVPR_2021_paper.html) (CVPR 2021)
2. (OadTR) [OadTR: Online Action Detection with Transformers](https://arxiv.org/abs/2106.11149) [code](https://github.com/wangxiang1230/OadTR) (arxiv 2021)
3. (LSTR) [Long Short-Term Transformer for Online Action Detection](https://arxiv.org/abs/2107.03377) (arxiv 2021)
