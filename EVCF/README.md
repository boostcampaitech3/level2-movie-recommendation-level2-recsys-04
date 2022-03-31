# Enhancing VAEs for Collaborative Filtering: Flexible Priors & Gating Mechanisms

### 먼저 utils/loda_data.py 들어가셔서 데이터 경로를 본인의 pro_sg로 다 바꿔주셔야 합니다.(저희 Multi-VAE에서 생성한 파일들과 동일하니 그 경로로 설정해주시면 됩니다)  
### python experiment.py하면 제 기본 세팅으로 훈련이 됩니다.  
### 훈련된 모델로 Inference 하시고 싶으시면 python experiment.py —inference True하시면 됩니다.(제 기준 input/submission폴더에 evcf_test.csv로 제출파일이 형성됩니다)
### 리더보드 기준 valid 없이(n_heldout=0)한 모델이 최고여서 이 모델 훈련 과정에서도 valid 관련 코드를 주석처리 해두었습니다(원하시면 푸셔도 됩니다)


This is the source code used for experiments for the paper published in RecSys '19:  
"Enhancing VAEs for Collaborative Filtering: Flexible Priors & Gating Mechanisms"    
(arxiv preprint: https://arxiv.org/abs/1911.00936, ACM DL: https://dl.acm.org/citation.cfm?id=3347015)

An example of training a hierarchical VampPrior VAE for Collaborative Filtering on the Netflix dataset is as follows:
`python experiment.py  --dataset_name="netflix" --max_beta=0.3 --model_name="hvamp" --gated --input_type="binary" --z1_size=200 --z2_size=200 --hidden_size=600 --num_layers=2 --note="Netflix(H+Vamp+Gate)"`

### Requirements
Requirements are listed in `requirements.txt`

### Datasets
Datasets should be downloaded and preprocessed according to instructions in `./datasets/`

### Acknowledgements
Many of our code is reformulated based on https://github.com/dawenl/vae_cf and https://github.com/jmtomczak/vae_vampprior
