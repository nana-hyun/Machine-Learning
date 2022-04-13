# 4.1. Decision Boundary

![image](https://user-images.githubusercontent.com/101063108/159625965-f04f68bc-5812-4f5b-adfd-cdc9f582d21f.png)

초록 선과 빨간 선이 교차되는 지점을 xm 이라 하면, decision boundary는 xm이 된다.

선형적인 함수와 곡선의 함수 중 곡선의 함수가 더 유용하게 사용되는 이유는 risk가 적기 때문에, S-curve (a.k.a Sigmoid function)을 사용해야 한다.

![image](https://user-images.githubusercontent.com/101063108/159626481-908caceb-4847-4aa2-9b52-3b0e35c7d987.png)

예전에 예시로 들었었던, credit 정보를 예시로 설명해보자.

class C를 attribute A15를 이용해 예측할 수가 있는데 아래 표를 보면 알 수 있다.

![image](https://user-images.githubusercontent.com/101063108/159626978-6b83e0ea-603f-4a27-800a-c7f677074cf9.png)

C는 0(credit을 받지 못함) 1(credit을 받음)로 구분되어 discrete하다고 할 수 있다. 

첫번째와 두번째의 차이점은, 첫 번째의 경우 X에 A15값이 그대로 들어가는 반면, 두 번째의 경우  X에 A15에 log를 취한 값이 들어간다.

여기서 회색 선을 decision boundary라고 잡아보았는데, log를 취하게 되면 데이터들을 직관적으로 볼 수 있고 판단하기 쉬워진다.

그렇다면 더 정확하게 decision boundary를 잡으려면 어떻게 해야할까?

선형회귀를 이용하는 방법과 logistic function을 이용하는 방법이 있다.

아래 빨간 점들의 자취가 선형회귀를 이용하여 피팅한 선형 함수이다. 이러한 경우의 문제는, 0과 1사이의 값을 가져야하는 확률론을 위반하고, 표본에 대해 느린 반응을 보인다는 점이다.

![image](https://user-images.githubusercontent.com/101063108/159645870-f38f78c5-c62e-43f0-b268-764541c7d057.png)
![image](https://user-images.githubusercontent.com/101063108/159645938-30e4c0a7-a171-451e-a7bf-051ae152d3fa.png)

logistic function을 이용해 피팅할 경우, 확률론에 위배되지 않으며, decision boundary 근처에서 빠른 반응을 보인다는 것이다.

왼쪽의 그림은 log를 취해 더 보기좋게 만든 것인데, logistic function이 앞서 보았던 s-curve, 즉 sigmoid fuction의 특별한 케이스임을 알 수 있다.

회색 부분이 decision boundary가 되는데, linear function일 때의 decision boundary와 logistic function일 때의 decision boundary가 다름을 알 수 있다.

* linear function의 경우

    decision boundary가 왼쪽으로 치우쳐 있으며, false case(0)인 경우, 대부분의 표본이 맞게 분류되지만, true case(1)인 경우 왼쪽 끝 부분을 제외하고는 잘못된 분류가 된다.
    
* logistic function의 경우

    decision boundary가 비교적 가운데에 위치하며, 각 case에서 분류가 잘못된 경우도 존재하지만, 비교적 합리적이다.
    

# 4.2. Introduction to Logistic Regression

## Logistic Regression

**Sigmoid function**

* 유계되어 있음 (bounded)
* 미분가능
* real function
* 모든 real input에 대해 정의되어 있음
* 양의 도함수를 가짐 (증가하는 함수)

![image](https://user-images.githubusercontent.com/101063108/159652212-2992fee3-35d8-482d-a7fb-3fb002a7d7e7.png)


**Logistic function**

![image](https://user-images.githubusercontent.com/101063108/159652137-9dc518e5-4b31-46ba-8285-7ff4dde5262e.png)

logistic function은 0과 1에 유계되어 있으며, 이의 역함수를 logit function이라고 한다.

logistic function의 함수식은 다음과 같다.

![image](https://user-images.githubusercontent.com/101063108/159652249-f04835d1-a30c-4bf2-85f6-af3b553fc59b.png)

logit function의 함수식은 다음과 같다.

![image](https://user-images.githubusercontent.com/101063108/159652984-f4ce718d-2fbe-45b7-809b-1482f20d97ad.png)

    인구 증가 그래프에 유용하게 쓰임

이러한 logistic function은 도함수를 계산하기 쉽다. 즉, optimization을 할 때, 극값이 0인 지점의 계산이 쉬워지기 때문에 자주 사용된다.

## Logistic Function Fitting

![image](https://user-images.githubusercontent.com/101063108/159662913-af977fd7-e70d-441b-9172-3a2f8cbabe4e.png)

첫번째의 경우는 logitic function이다. logitic function의 경우, x domain이 0~1의 값을 가지게 되고, 이는 확률 p로 작성할 수 있다.

![image](https://user-images.githubusercontent.com/101063108/159664639-3b179d60-0e09-4703-a04e-525cf4646400.png)

이 함수를 선형변환을 통해 더 보기좋게 만들어 줄 수 있는데, 이 때 계수 a를 통해 압축 또는 확장이 가능하고, 상수 b를 통해 이동이 가능하다. 

ax + b는 선형 함수의 모양이기 때문에, X와 θ를 이용해 표현 할 수 있다. 이는 선형 회귀의 식과 유사하므로, 이를 이용해 피팅할 수 있다.

![image](https://user-images.githubusercontent.com/101063108/159668799-184ffb1d-a88c-43ee-a852-a70201fb9f2f.png)

P(Y|X)를 선형회귀를 이용해 피팅을 하고자 하는데, 이 때 이 확률을 모델에 더 넣어 logistic함수의 모양을 활용하는 logistic regression이 된다.

## Logistic Regression

logistic regression은 이항 또는 다항 결과를 예측하기 위한 확률론적 분류법이다.

* logistic 함수에 조건부 확률을 피팅

Bernoulli experiment

![image](https://user-images.githubusercontent.com/101063108/159692642-a4c1a6a4-c35b-4731-a821-fe394c7da424.png)

logistic function형태로 모델링, 

![image](https://user-images.githubusercontent.com/101063108/159692794-59a08cbd-4d91-4b27-9261-ff59123a88b3.png)

여기서 ![image](https://user-images.githubusercontent.com/101063108/159692855-5f419fe3-6775-4810-805a-372195789538.png)
로 분모, 분자를 나누면 아래와 같이 logistic function 형태가 된다.

결국, 목표는 θ를 찾는 것이 된다. θ를 어떻게 배우는가?

# 4.3. Logistic Regression Parameter Approximation 1




