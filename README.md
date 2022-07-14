# ganspace
Implementing GANSpace(NeurIPS 2020) by PyTorch

inversion을 통해 찾은 w space상에서 PCA를 통해 component를 찾는 방식. 이 comnent의 방향이 editing direction임.

따라서 찾은 component 방향으로 latent vector를 변화시키면 이미지가 editing 된다.

장점 → unsupervised 방식이기에, editing시 별도의 훈련과정 필요하지 않음. + 어떤 데이터셋을 사용하든 GENERAL 한 solution이 될 수 있음. 

단점 → 시간이 많이 소요됨 (찾은 component가 어떤 attribute를 변화시키는지 직접 눈으로 확인해야 함 + 변화시킨 latent vector를 어느 styleGAN2의 어느 layer에 넣어주느냐에따라 결과가 달라지기에, 이를 실험적으로 찾아야 함)


코드분석

- 논문을 읽은 후에는 이미지 여러장에 대해서 inversion을 진행한 뒤, 뽑아낸 latent vector를 가지고 PCA를 진행하는 줄 알았는데, 훈련시 사용한 normal 분포에서 z를 샘플링을 한 다음 PCA를 진행함. (W space상에서 direction을 찾는 경우, z에서 랜덤하게 추출한 뒤 mapping network를 통과한 뒤에 PCA진행)
- GANSpace 깃헙 코드는 styleGAN이나 논문에서 실험한 모델을 중심으로 짜여져있어서 3D 모델에 그대로 사용하기는 어려움. → 직접 코드를 짰음

코드실행결과

- ganspace 공식 깃허브에서 pca estimator만 그대로 가져오고, 논문보고 구현함. FFHQ StyleGAN사용
- 논문에서도 언급했듯이 몇번째 pca components를 수정하느냐, 어떤 layer에만 w를 넣어주느냐에 따라 결과가 달라짐. 다 일일히 확인해봐야함.. + 얼마만큼 direction 조절할지도 정해야함
- 논문에서는 10^6개 샘플링을 통해 pca를 진행했지만, 연구실 서버로 돌려보니 10^4로 돌리는 게 최대였음. (10^5부터 OOM 뜸)

(v9, 0-6) [-2, 2] → rotation 확인(논문에 나와있는 방법임)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/003d0cb0-fe2a-4148-a2e2-a75f64177696/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0bba3c45-6f37-4501-95a3-e7bfb722726c/Untitled.png)

(v18, 0-6) [-2, 2] → smile ^^
