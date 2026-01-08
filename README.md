# ImageClassificationGZIC
**ResNet50 + Vision Transformer (ViT) 混合模型實現華南理工大學廣州國際校區 (GZIC) 六類物品影像分類**

使用自建資料集訓練一個結合 ResNet50 和 ViT 的模型，對校園相關六類物品進行影像分類。  
學生需自行收集資料、訓練模型，並提交測試腳本給助教評分。
## Task
In this project, you are required to do a classification task on 6 types of objects related to Guangzhou International Campus, SCUT.  

|<img src="./img/1.jpg" width="800" /> |<img src="./img/2.jpg" width="800" /> |<img src="./img/3.jpg" width="800" /> |<img src="./img/4.jpg" width="800" /> |<img src="./img/5.jpg" width="800" /> |<img src="./img/6.jpg" width="800" /> |
|:----:|:---:|:----:|:----:|:---:|:----:|
|Bell tower|Library|School logo|Liyujun mascot|Mingcheng mascot|Junde mascot|


Use all the methods and tricks, including you learned in the lectures or from other resources, to train your own model with your own dataset.   

1. You should create your own dataset based on the examples. The examples of the test set are released and you should create your own dataset referring to the examples by taking photos or collecting images from the Internet. Do not use the released examples in the training!  
2. You should submit your model and test file to TA then obtain a test result. The test set will be kept by the TA in order to avoid possible cheating by directly training the model on the test set. You should submit a test  script with your model to TA then the TA cantest your result. The template of the test script is released with the data examples and you have 3 chances to submit the model. The data in the test set will be similar to the given examples and please validate your test script on the examples first.  
3. A project report within 5 pages and an oral presentation with slides should be completed.
   
總結：需要自己準備資料集、訓練模型，提交模型及代碼給TA獲取準確率結果，一共三次提交機會。
## Dataset
透過拍攝與爬蟲蒐集  
[Download](https://1drv.ms/u/c/585289ea0ef7a626/ETDIBRBdxuBBgQv6jCciBfcBI0CcsItDbxDitjKXL5GvTQ?e=t3P13T)

## 模型結構
混合模型：ResNet50 作為特徵提取器 + Vision Transformer (ViT) 進行分類。
- 使用轉移學習（pre-trained ResNet50 + ViT）。
- 資料增強：隨機裁剪、翻轉、顏色抖動等。


## 腳本說明
### `train.py`
主訓練腳本，使用方法： 
```
python train.py --data_dir ./dataset --epochs 50 --batch_size 32 --lr 0.001
```
### `test_script.py`
提交給助教測試用的腳本模板
### `model_skeleton.py`
模型結構（ResNet50 + ViT）  
### `sougou_scraper.py`
搜狗圖像爬蟲

## Essay
[View](./Deep%20Learning%20Project%20on%20Image%20Classification(github).pdf)

## Slides
[View](./Deep%20Learning%20Project%20on%20Image%20Classification%20PPT%20(github).pdf)

## Team Member Contribution
|Teammate|Individual Contributions|
|---|---|
|A |Conducted preliminary literature review; Developed data preprocessing  scripts  (including  but  not  limited  to  data augmentation and background removal using masks) Built, selected   and   trained   models;   Designed   and   conducted experiments; Made 9 charts for Report; Assisted in refining the  report  and  presentation  slides;  Took  charge  of  the presentation and Q&A session.|
|B (repository maintainer)|Coded  a  Web  Scraper  for  Sogou;  Took,  collected  and annotated over 1600 photos; Translated, revised and added content to Report; Made 1 chart and 6 formulas for Report; Proofread and added content to PowerPoint; Edited and added content to speech script.|
|C |Took, collected and annotated approximately 300 images; Drafted  the  initial  Chinese  version  of  the  report,  created presentation slides and the speech script; Making 1 chart for Report; Presented the content using the slides.|

