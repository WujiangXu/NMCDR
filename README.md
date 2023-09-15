# NMCDR
This repository is the official code for ICDE2023 paper ["**Neural Node Matching for Multi-Target Cross
Domain Recommendation**"](https://arxiv.org/abs/2302.05919).  
By **Wujiang Xu, Shaoshuai Li, Mingming Ha et al.**  
## Data Processing 
1. Download Amazon data from the url "http://jmcauley.ucsd.edu/data/amazon/index_2014.html". 
2. Process the data by  
>python3 process.py #1st step  
>python3 filter_dataset.py #2nd step  

## Train the Model 
>python3 train_matching.py

## Future Work
More detailed parameters descirption.

## Citation

If you found the codes are useful, please cite our paper.

      @inproceedings{xu2023neural,
      title = {Neural Node Matching for Multi-Target Cross Domain Recommendation},
      author = {Wujiang Xu, Shaoshuai Li, Mingming Ha, Xiaobo Guo, Qiongxu Ma, Xiaolei Liu, Linxun Chen and Zhenfeng Zhu},
      booktitle = {	The IEEE International Conference on Data Engineering 2023 (ICDE2023)},
      year = {2023}
      }




## Contact us 
Please feel free to contact us with the email to W. Xu "xuwujiang dot xwj at mybank dot cn" or "swustimp at gmail dot com".
