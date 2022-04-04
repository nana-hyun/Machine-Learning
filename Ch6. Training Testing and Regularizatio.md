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

에러는 두가지 경우에서 발생할 수 있다.

1. Approximation

approximation하면서 발생하는 에러들.

2. Generalization

앞으로 올 데이터에 대해서 발생하는 에러들.

![image](https://user-images.githubusercontent.com/101063108/161492194-1071b910-644e-4e26-81ad-adf9f8cafa50.png)

우리가 추정하는 에러인 Eout은 Ein과 Ω의 합보다 작거나 같다라고 말할 수 있으며,

Ein은 approximation을 할 때 발생하는 에러이고, Ω은 관찰의 variance에 의해 발생되는 에러이다.

본격적으로 식을 전개하기에 앞서 몇 가지 심볼들에 대해 정의를 하자.

* f : 우리가 배우는 target function
* g : 우리가 ML을 통해 배운 function
* g(D) : dataset D를 사용하여 배워진 function
* D : 실제에서 가져온 이용가능한 dataset
* ![image](https://user-images.githubusercontent.com/101063108/161495211-fc80f9e9-46d9-44b6-8390-c1665c409805.png) : D를 무한대로 해서 가져온 hypothesis의 평균

![image](https://user-images.githubusercontent.com/101063108/161495422-def28159-9e09-4442-b92e-eb73e18f3d6c.png)

## Bias and Variance

Bias : 편향 Variance : 분산

단일 dataset D에 대해서 Eout을 계산하면 다음과 같다.

![image](https://user-images.githubusercontent.com/101063108/161497271-6bc72524-81d0-425a-8b82-f9c636c616fe.png)

dataset D의 무한대에 대해서 기대된 에러값은 다음과 같이 작성될 수 있다.

![image](https://user-images.githubusercontent.com/101063108/161497512-4d48ba3e-e320-4b4c-a3ae-0bc57f71ab80.png)

여기서 ![image](https://user-images.githubusercontent.com/101063108/161497764-cccb507b-86c6-4584-b6c9-300297628c5d.png)는 데이터에 대해서 expected error를 여러번 계산하는 것을 의미한다.

![image](https://user-images.githubusercontent.com/101063108/161497957-ae8405d2-bf3a-4e51-9ae2-a29b57b60cbe.png)
 이 식을 조금 간단히 만들어 보자면, 다음과 같이 전개할 수 있다.
 
 ![image](https://user-images.githubusercontent.com/101063108/161498100-293a88a1-95ed-4a86-a23a-64f6a81c8d8b.png)

이때 ![image](https://user-images.githubusercontent.com/101063108/161498239-e18fbdf6-b883-4f2f-871c-7e68a9a19fde.png) 이 부분은 0 값을 가지게 된다. : ![image](https://user-images.githubusercontent.com/101063108/161498354-21bd6941-89a4-4b38-bce8-a7ad66666b34.png)의 정의가 무한대의 샘플링을 통해 expected한 값과 동일하기 때문이다.

![image](https://user-images.githubusercontent.com/101063108/161498722-cae711c5-5871-4c9b-ba28-4773afe6bd32.png)
 
 따라서 위와 같이 정리할 수 있다.
 
 위의 식에서
 
 ![image](https://user-images.githubusercontent.com/101063108/161498818-f711639b-6974-433d-819c-d64eb392d136.png)


라고 정의해보자.

variance는 우리가 가진 단 하나의 dataset으로 만든 ML 모델과 앞으로 들어올 모든 dataset에 대한 average hypothesis의 차이에서 오는 에러

bias는 우리가 가진 모델의 한계점에 의해 생길 수 있는 에러 : 아무리 좋은 dataset을 가져와도 우리가 만든 모델이 true function이 될 수 없음.

* reducing variance : 더 많은 데이터를 모으기

* reducing bias : 더 복잡한 모델 만들기

그러나 variance와 bias는 서로 tradeoff이다. : bias가 감소하면 variance가 증가, bias가 증가하면 variance가 감소

-> Bias and Variance Dilemma

# 6.3. Occam's Razor

![image](https://user-images.githubusercontent.com/101063108/161517711-6ab2b8a9-1568-4e6e-8019-4cf5b91cb0ce.png)

true functiondl sin함수라고 하자.

D={two points|point=(x,sin(2 * pi * x)),0<=x<=1}

two g(x)

* zero degree : dark grey line
* one degree : light grey line

two ![image](https://user-images.githubusercontent.com/101063108/161518442-c22e1b11-3a97-4b03-b35d-168e66772f6a.png)(x)

* zero degree : red line
* one degree :  green line

constant line의 경우, stable하나(low variance) 1차 line보다 true function에 덜 fitting되어 있다고 할 수 있으며,(high bias) 

1차 line은 만일 sample point가 다르게 잡힌다면, error가 많이 생기는 1차 line이 나올 수 있다.(high variance) & (low bias)

![image](https://user-images.githubusercontent.com/101063108/161519990-cf363466-7e93-440b-a42a-8281ade356e9.png)

![image](https://user-images.githubusercontent.com/101063108/161521081-e88fd6ac-a250-4e46-a54e-de4747ebbccc.png)

* complex model : higher variance & lower bias
* simple model : lower variance & higher bias

이 둘의 밸런스를 유지하는 것이 필요하다.

![image](https://user-images.githubusercontent.com/101063108/161501111-a80e2bd6-cbac-4df3-b64d-1e03de292f09.png)

**Occam's Razor**

competing(prediction에서 비슷한 error 가지는 것) hypotheses에서 fewest assumption이 선택된다는 것.

fewest assumption이란? 덜 복잡한 모델

즉, approximation을 할때 같은 error가 주어지면, 더 간단한 모델이 선택되는 것이다.

