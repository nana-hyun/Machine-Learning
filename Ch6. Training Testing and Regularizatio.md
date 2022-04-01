# 6.1. Over-fitting and Under-fitting

여기서 이제 우리는 우리가 만든 모델이 정확하게 작동하는지 확인해야 한다.

만일 100개의 case 중 하나만 true case이고 나머지 99개는 false case라고 하자. 여기에 그냥 무조건 false만 내보내는 모델이 있다면, 그 모델은 99%의 accuracy를 가지지만, 이것이 제대로 된 모델이라고 말할 수 없을 것이다.

* the validity of accuracy
1. 이렇듯 다양한 accuracy에 대해 명확한 definition을 알아야함.
2. precision/recall,F-measure에 대해서는 어떻게 되는지 계산.

* the validity of dataset
1. 극단적인 dataset이 실생활에서 일어날 수 있는가? test되는 dataset이 올바른지 알아야함.
2. spam이 얼마나 있는지, 어디서 모았는지..

**Training and Testing**

* Training

parameter inference를 하기 위해 쓰임.

prior knowledge, past experience에도 사용될 수 있다.

ML은 주어진 데이터의 분포가 변하지 않을 때까지만 사용할 수 있다.

dataset의 분포가 바뀌면, 거기에 맞춰서 다시 training 하는 과정이 필요하다.

ML이 미래에 잘 적용이 안되는 경우 : 여러가지 도메인이 바뀌는 경우, 데이터가 계속 변하고 있을 때. 현재의 도메인이 충분한 variance에 대해 설명하고 있지 않을 때

전체 데이터 중 일부는 Training하는데 사용하고, 다른 부분은 testing하는데에 사용되는데에 쓴다.

* Testing

만일 destortion된 decision boundary가 설정되었을 때, 우리는 testing을 통해 distortion 된 것을 확인할 수 있다.

따라서 dataset을 training과 testing으로 나눠서 적용했을 때 distortion된 모델들을 검출하고 정확성을 확인할 수 있다.

![image](https://user-images.githubusercontent.com/101063108/161198021-a95d7fb4-b80d-4c1a-ae32-2c6fc1905754.png)

**Over-Fitting and Undr-Fitting**

아래는 N개의 data point에 단순한 polynomial regression을 피팅한 상황이다. Y=F(x)라고 한다면 차수가 정해지지 않았으므로, 선형적이 될 수도 비선형적이 될 수도 있다.

아래 3개의 경우가 있다고 해보자.

![image](https://user-images.githubusercontent.com/101063108/161198532-3bef9f90-d044-4284-919d-cd9b94a65907.png)

어떤게 정확할까? : 두번째 것이 가장 적절하다고 볼 수 있다.

첫번째의 경우는 맨 앞부분이 training이 덜 된 모습을 볼 수 있다. 조금 더 피팅이 필요하다 -> underfitting

세번째의 경우는 point마다 피팅을 하려 노력한 모습. 에러가 아주 작지만, 이 트렌드에 맞춰서 앞으로 올 데이터에 대해서는 두번째보다 에러가 많을 수 있다. -> overfitting

우리는 차수를 늘림으로써, 모델을 complex하게 만들 수 있다. 무조건 complex한 것이 더 좋은 모델이 아니기 때문에, complex를 developing하는 것을 멈추는 순간이 와야한다. 

너무 specialized 되어있고, complex한 것은 general한 data가 왔을 때 제대로표현하지 못할 수 있다.

complex는 조금 떨어지더라도 앞으로 올 general한 데이터에 대해 잘 말할 수 있다.

# 6.2. Bias and Variance

