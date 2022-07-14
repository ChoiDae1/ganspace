# GANSpace
Implementing GANSpace(NeurIPS 2020) by PyTorch

### 논문 간략 리뷰

훈련시 사용한 w space상에서 PCA를 통해 component를 찾는 방식. 이 component의 방향이 곧 editing direction임.<br>
따라서 찾은 component 방향으로 latent vector를 변화시키면 이미지가 editing 된다는 원리.<br>
<strong>장점:</strong> unsupervised 방식이기에, editing시 별도의 훈련과정 필요하지 않음. + 어떤 데이터셋을 사용하든 GENERAL 한 solution이 될 수 있음. <br>
<strong>단점:</strong> 시간이 많이 소요됨 (찾은 component가 어떤 attribute를 변화시키는지 직접 눈으로 확인해야 함 + 변화시킨 latent vector를 어느 styleGAN2의 어느 layer에 넣어주느냐에따라 결과가        달라지기에, 이를 실험적으로 찾아야 함)


### 원본 코드분석-><a href="https://github.com/dmesh-io/ganspace/tree/3b833f927c603890acb7e15aeded4ae06e76347f">link</a>

- 논문을 읽은 후에는 이미지 여러장에 대해서 inversion을 진행한 뒤, 뽑아낸 latent vector를 가지고 PCA를 진행하는 줄 알았는데, 훈련시 사용한 normal 분포에서 z를 샘플링을 한 다음 PCA를 진행함. (W space상에서 direction을 찾는 경우, z에서 랜덤하게 추출한 뒤 mapping network를 통과한 뒤에 PCA진행)
- 원본 코드는 styleGAN이나 논문에서 실험한 모델을 중심으로 복잡하게 짜여져있어서 곧바로 활용하기는 어려움. → 직접 코드를 짰음(image_manipulation.py 참고바람)


### 코드실행결과

- ganspace 공식 깃허브에서 pca estimator만 그대로 가져오고, 논문보고 구현함. FFHQ pretrained StyleGAN사용
- 논문에서도 언급했듯이 몇번째 pca components를 수정하느냐, 어떤 layer에만 w를 넣어주느냐에 따라 결과가 달라짐. 다 일일히 확인해봐야함.. + 얼마만큼 direction 조절할지도 정해야함
- 논문에서는 10^6개 샘플링을 통해 pca를 진행했지만, 연구실 서버로 돌려보니 10^4로 돌리는 게 최대였음. (10^5부터 OOM 뜸)

(v9, 0-6) [-2, 2] → rotation 확인(논문에 나와있는 방법임)

![image](https://user-images.githubusercontent.com/95220313/179071958-8132f317-5737-418c-a084-29bcef8c4cfc.png)

(v18, 0-6) [-2, 2] → smile 
![image](https://user-images.githubusercontent.com/95220313/179072048-258f7181-8d50-49a5-964d-76d656f21f48.png)
