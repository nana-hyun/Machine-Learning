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

# 6.4. Cross Validation

우리는 target function으로부터 관찰되는 무한대의 샘플을 얻을 수 없다.

그렇다면 우리는 이런 무한대의 sampling을 흉내내야 하는데, 앞전에 말했던 bias와 variance tradeoff에서 무한대의 dataset들을 가정하고 average hypothesis를 만들었었다.

수많은 dataset들을 가지고 무한대인 것 처럼 계산을 해주어야한다.

이때 쓰이는 유용한 방법이 N-Fold Cross Validation (K-Fold Cross Validation) 이다.

주어진 instance의 집합을 N개의 subset으로 나누고, (N-1)개의 subset들을 training하는데 사용하고, 1개의 subset을 testing에 사용한다.

![image](https://user-images.githubusercontent.com/101063108/161527937-c6d2647d-fa80-4a28-b994-a716ff68b23c.png)

각 instance 하나하나를 나누고, 하나만 빼고 training하고, 나머지 하나를 testing 한다면 data instance의 개수만큼 진행할 수 있다.

이것이 가장 특별한 케이스인 **LOOCV** 이다.

* Leave One Out Cross Validation

N-fold cross validation의 가장 극단적인 케이스로 볼 수 있다.

# 6.5. Performance Metrics

우리는 target function f(x)를 모르기 때문에, 우리는 average hypothesis인 ![image](https://user-images.githubusercontent.com/101063108/161665807-877d88b5-ff1a-492e-9e9d-fcfb8a1dd3c8.png) 를 계산할 수 없다.

performance measure로서 bias와 variance를 사용할 수 없다.

다른 performance measure들

* Accuracy = (TP+FN) / (TP+FP+FN+TN)
* Precision and Recall
* F-Measure
* ROC curve

![image](https://user-images.githubusercontent.com/101063108/161666127-48303740-bf69-4a29-84cd-425d5c93aef6.png)

* TP (True Positive) : 실제로 true인 case를 true라고 추정
* FN (False Negative) : 실제로 true인 case를 false라고 추정
* FP (False positive) : 실제로 false인 case를 false라고 추정
* TN (True Negative) : 실제로 false인 case를 true라고 추정

![image](https://user-images.githubusercontent.com/101063108/161666871-dc44cf26-33e6-4104-b9d5-7c7a08d82091.png)

*different goals

1. Spam filters

스팸 메일을 분류할 때, 스팸메일이 스팸함으로 들어가지 않는 것보다는 중요한 메일이 스팸함으로 들어가는 것이 더 치명적이다.

즉, 우리는 스팸으로 분류된것을 확인했을 때 스팸메일만 있어야 하는것이다. False Positive의 case는 중요한 메일이 스팸함으로 분류되었을 때를 의미한다.

이 FP를 우선적으로 줄여야하며, 이를 위해 초록색 박스 부분을 measure해야한다.

이 부분을 **Precision** 이라고 한다. 

***TP/(TP+FP)***

positive라고 예측한 것 중 진짜 true일 확률

2. CRM : classifying VIP customer

VIP 고객들을 절대 빼놓으면 안되는 상황에서, 수많은 고객 중 해당 VIP 고객만을 뽑아내는 일은 쉽지않을 것이다.

이 경우 고객을 뽑을 때, VIP 고객이 아닌 사람이 명단에 포함되는 것보다, VIP 고객이 명단에 포함되지 않는 것이 더 치명적이다.

False Negative는 VIP고객이 명단에 포함되지 않는 경우를 의미한다. 이 FN을 줄여야하고 이를 위해 보라색 박스 부분을 measure해야한다.

이 부분을 **Recall** 이라고 한다.

***TP/(TP+FN)***

true인 case 중에서 positve로 분류되는 확률

### F-Measure

이러한 precision과 recall이 자주 사용되는 방법이지만, 다음과 같은 문제가 발생할 수 있다.

가장 안전한 스팸필터 = 항상 스팸이 아니라고 분류

VIP고객에 연락하는 것이 보장된 고객 필터 = 항상 VIP라고 분류

precision과 recall의 밸런스를 맞춰주기 위해 F-Measure의 방법을 사용한다.

![image](https://user-images.githubusercontent.com/101063108/161670663-722b649f-786d-401b-af77-4174929e0b36.png)

# 6.6. Definition of Regularization

![image](https://user-images.githubusercontent.com/101063108/161920386-83e05e1a-8f1c-42d0-a989-1a31020ca237.png)

![image](https://user-images.githubusercontent.com/101063108/161921957-f8fabaaa-96ad-4dca-9c31-0f35c269ad6a.png)

이런식으로 잘못된 점을 샘플링했을 때, 에러가 커질 수 있다. 이렇게 되면 오히려 위에서 봤던 상수함수가 더 좋을 수 있다.

따라서 regularization을 해주게 되는데, regularization의 concept은 perfect fit을 포기한다는 것이다. 

training accuracy를 줄임 - 앞으로 올 test 데이터들에서의 potential fit을 증가시킴

complex한 모델의 경우, bias가 감소하는 경향을 보인다. 그러면 overfitting이 되는데, 이를 막기 위해, 두가지 방법이 있을 수 있다.

1. complex하지 않은 모델을 만들자!

complex한 모델을 만들지 않고, 상수 함수같이 단순한 모델들을 만들면 된다. 그러나 이 경우, bias가 증가하여 에러가 증가할 수 있다.

2. complex한 모델을 만들되 둔감하게 만들자!

bias 에러가 증가하는 것을 막기 위해, complex한 모델이지만 데이터에 둔감하게 만들어보자. 이 경우, perfect fit은 못하지만, 데이터의 경향성은 따르게끔 만들 수 있다.

-> 이것이 regularization의 목표 : 모델을 너무 민감하게 만들지 않는 것.

**Formal Definition of Regularization**

regularization은 regression에 대한 제약이라고 할 수 있다.

![image](https://user-images.githubusercontent.com/101063108/161941339-ef1799b3-6469-4214-bfed-fbfcced1256a.png)

식에서 앞쪽 부분은 에러를 줄여주는 training의 정확도를 의미하는 부분이고 뒤는 regularization term이다. 

regularizaiton term은 lambda와 가중치로 구성이 되는데, lambda의 값이 커지면, 가중치의 값이 작아진다. 따라서 lambda가 0이 되면, 일반적인 선형 회귀 모형이 된다.

위 두개의 식의 차이는 가중치의 제곱값이 들어가는지, 가중치의 절댓값이 들어가는지의 차이이다.

![image](https://user-images.githubusercontent.com/101063108/161943604-63eaaa7f-9a91-4c8a-9ced-75829b30ac30.png)

* L1 Regularization == Lasso regularization

밑의 식에 해당된다. 그림을 보면 알 수 있듯이, 마름모 꼴의 모양이 나오고, 몇몇 파라미터들을 0으로 만들어 주어 변수 선택이 가능하다는 특징이 있다.


변수를 선택함으로써 모델을 단순하게 만들어줄 수 있다. 미분은 불가능하다.

* L2 Regularization == Ridge regularization

위의 식에 해당된다. 원 형태의 제약조건을 그릴 수 있으며, 이는 어디든지 접할 수 있기 때문에, 많은 파라미터들이 0이 아닌 값을 가지고, 변수 선택이 불가능하다.

적절한 가중치 배분이 가능하며, 미분을 통해 구할 수 있으므로 최적화 기술들을 잘 활용할 수 있다는 장점이 있다.

# 6.7. Application of Regularization

ridge regularization의 아이디어를 linear regression에 적용해보자.

![image](https://user-images.githubusercontent.com/101063108/162034712-047c6ef0-9b68-425a-b030-b94214f48855.png)

w로 편미분한 값을 0이라고 하고 계산하여 적절한 w의 closed form을 구할 수 있다.

![image](https://user-images.githubusercontent.com/101063108/162035020-fdc9c9f2-7053-4cb6-94a2-2af84edfb258.png)

![image](https://user-images.githubusercontent.com/101063108/162035058-61cfeb93-0765-426a-b0dd-860bba277508.png)

아래는 이제 regularization을 적용하기 전과 후의 그래프이다.

![image](https://user-images.githubusercontent.com/101063108/162035243-48d36215-742e-4cce-aeb6-6b24d0f90140.png)

첫번째에서 regularization(lambda = 1)을 통해 두번째 그래프로 그려졌는데, bias는 약간 증가했지만, variance가 훨씬 더 많이 감소한 것을 알 수 있다.

-> 에러를 감소하는 결과를 기대할 수 있다.

위에서는 lambda가 1일때의 결과를 보여주었는데, lambda 값이 변하면 regularization은 어떤 결과를 만들어낼까?

![image](https://user-images.githubusercontent.com/101063108/162035775-966133b0-5cc1-42f8-a8bf-7af41bc1faaf.png)

* lambda = 0 (너무 낮은 lambda를 가질 때)
    * 너무 높은 variance를 가진다.
    * regularization을 적용하지 않은 것과 결과가 같다. 
* lambda = 100 (너무 높은 lambda를 가질 때)
    * 너무 낮은 variance를 가진다.
    * complex하지 않은, 너무 단순한 모델을 만들며, 1차함수 모델이 상수함수처럼 보인다.

우리는 적절한 lambda값을 찾아야하고, 이를 위해 실험을 여러번 시도해보아야한다.

**Regularization of Logistic Regression**

regularization은 logistic regression에도 적용할 수 있다.

* logistic regression에서의 ɵhat

![image](https://user-images.githubusercontent.com/101063108/162038324-5eda1880-847c-47d1-934f-f7f073511c13.png)

우리는 closed form을 찾고 적절한 ɵ의 형태에 접근할 수 있다.

![image](https://user-images.githubusercontent.com/101063108/162037366-9d832e24-9e59-4ab9-9918-0894d0b9bcde.png)

![image](https://user-images.githubusercontent.com/101063108/162037402-cca0ec95-a49f-4a5b-9ab4-9fa733d83a6e.png)

regularization 부분까지 감안해서 apporximate 해야한다.

**Regularization and SVM**

![image](https://user-images.githubusercontent.com/101063108/162039818-1190ef33-a5eb-4892-9417-4954f7a4ce8d.png)

예전에 SVM을 배울 때 봤던 PPT에서 가져와 보았다.

regularization과 SVM에서의 C는 유사하게 작동하는 부분이 있다.

soft-margined하다는 것은 regularization을 수행하는 것을 포함한다고 생각할 수 있다.

![image](https://user-images.githubusercontent.com/101063108/162040214-2715829d-435b-425b-9699-906dd92c1c2c.png)

따라서 우리는 이러한 C값을 위와 같이 적을 수 있다.

