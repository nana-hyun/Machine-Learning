# 2.1. Rule based machine learning overview

### definition of machine learning

* 경험 E에 의해서 배울 수 있다 
* 특정 과제 T를 수행을 할 수 있다.
* 경험 E이 늘면 과제 T를 더 잘 수행 P해낼 수 있다.

**More experience -> more thumbtack toss, more prior knowledge**

## A perfect world for rule based learning

![image](https://user-images.githubusercontent.com/101063108/157004159-ea13f7da-8e9a-4cae-8162-a34b5b45c420.png)

경험E : 여러 관측들

과제T : Temp, Humid, Wind, Water, Forecast -> EnjoySport

.

완벽한 세상에 대한 가정

> 관측의 오류는 없음. 일관적이지 않은 관측도 없음.

> 랜덤적인 요소가 없음.

> 시스템을 완전하게 설명할 수 있는 정보들을 다 수집함.


어떠한 factor가 가장 큰 영향을 미치는걸까?


### Fuction Approximation

다양한 정보들을 바탕으로 실제와 같은 결과를 만들어내는 것.  

Machnie Learning은 더 나은 근사 함수를 만들내려는 노력이다.

PAC learning과 비슷 : 낮은 오류를 만들어내는 것.

instance **X** : example -> 앞에서의 경우엔 4개의 instance가 있다고 말할 수 있다.

    * feature O: input값 <sunny, warm, normal, strong, warm, same>
    * label Y: 실제 판단의 결과 <Yes>

이 **X**를 여러개 모아놓은 것이 Training Dataset **D**

Hypotheses **H** : 이 함수라면은 현실과 동일하게 작동할 것이라는 가설 설정

    h1 : <Sunny, Warm,? ,? ,?, Same> -> Yes
    여러개의 hypotheses가 가능하다.

Target Function **c**

    진짜 함수, 알지 못하지만 목표로 하여 알아내야 하는 함수
    H를 c로 만들어주어야 한다.

![image](https://user-images.githubusercontent.com/101063108/157284223-f0dcc023-8c12-4079-959d-1e3915e931f1.png)

x1 : h1, h2, h3 모두 해당

x2 : h1, h2에만 해당

x3 : h1, h3에만 해당

    h1은 h2, h3에 비해 general하고, h2와 h3는 h1에 비해 specific하다고 말할 수 있는데,
    general하다면 instance space가 더 크고, specific하다면 instance space가 더 작다.


# 2.2. Introduction to Rule Based Algorithm

## Find-S Algorithm

![image](https://user-images.githubusercontent.com/101063108/157349881-8b152230-90ba-499f-8634-10f47dac1f03.png)

> Instance (positive한 사례들)
* x1 : <Sunny, Warm, Normal, Strong, Warm, Same>
* x2 : <Sunny, Warm, Normal, Light, Warm, Same>
* x3 : <Sunny, Warm, Normal, Strong, Warm, Change>

> Hypotheses
* h0 = <Ø, Ø, Ø, Ø, Ø, Ø>
* h1 : <Sunny, Warm, Normal, Strong, Warm, Same>
* h1,2,3 : <Sunny, Warm, Normal, ?, Warm, Same>
* h1,2,3, 4 : <Sunny, Warm, Normal, ?, Warm, ?>

![image](https://user-images.githubusercontent.com/101063108/157393138-933200b1-81a8-4b28-9239-8a80122c5c3e.png)

위와 같은 방식으로 특정한 hypothesis 찾고, function approximation을 해나가는 과정이 Find-S Algorithm이다.

## Version Space

다양한 가설들이 가능하나 딱 하나로 수렴하는 가설을 찾기는 어렵다. 가능한 가설들의 범위를 찾아야한다. 이러한 범위를 Version Space, **VS** 라고 한다.

* General Boundary, **G**
    VS에서 가장 일반적인 가설의 집합
* Specific Boundary, **S**
    VS에서 가장 구체적인 가설의 집합
* 모든 hypothesis, **h**는 다음을 만족한다.
    
    ![image](https://user-images.githubusercontent.com/101063108/157395883-8b44f317-a212-4710-a2ef-ff625ebe31f2.png)

![image](https://user-images.githubusercontent.com/101063108/157396036-736f9017-cebb-4b61-9025-6c61cca4596f.png)

> 가장 General하게는 Sunny 또는 Warm만으로 정할 수 있지만, Specific하게는 Sunny와 Warm, Strong이 포함된 가설을 세울 수 있다.

### Candidate Elimination Algorithm

가장 specific한 가설과 가장 general한 가설을 세운 후 점점 좁혀나가며 특정한 VS를 찾는 알고리즘.

가장 specific한 가설 - S : S0: {<Ø, Ø, Ø, Ø, Ø, Ø>}

가장 general한 가설 - G : G0: {<?, ?, ?, ?, ?, ?>}

![image](https://user-images.githubusercontent.com/101063108/157414110-ffc2bb38-6e74-432c-97a9-b2ef4c7b4cf6.png)

![image](https://user-images.githubusercontent.com/101063108/157433742-02490e48-82ab-4bd9-a3b1-50782973f4c6.png)

![image](https://user-images.githubusercontent.com/101063108/157433636-6cd95436-a096-4282-9328-4c857a75eea4.png)

S1 -> S2 : Humid가 Normal과 High일 때 둘 다 결과가 positive하므로 ?로 변경

-> G3 : Rainy or Cold or Change일때 negative한 결과. 고려해서 h세우기.

-> S4 : Water가 Warm과 Cool이고 Same과 Change일때 둘다 결과가 positive하므로 ? 변경

-> G4 : <?,?,?,?,?,Same>은 change일때 positive하므로 맞지 않음. remove.

중간에는 여러 가설들이 있고, 그 중 하나가 true function C일 것이라고 가정하고 배우는 것이 rule based learning 이다. 

새로운 instance가 들어오면 VS를 좁혀나갈 수 있는 방법이 있을 것이다.

* <Sunny, Warm, Normal, Light, Warm, Same>
이것을 위에 적용해본다면 General은 맞지만, Specific은 틀리게 될것이다.

이러한 경우에 경험, 사례가 충분하지 못하기 때문에 판단을 내리지 못할 수 있다. 만족시키는 가설의 개수를 세어보는 방법도 있을 수 있지만 해답이 되기는 힘들다. 그렇기 때문에 rule based learning은 여러 분야에 다양하게 쓰이기가 쉽지 않다.

따라서 '완벽한 세상'에서 잘 작동한다. candidate-elimination algorithm은 무한히 데이터가 들어올 때 true function으로 수렴가능하다.

앞서 언급했듯이 완벽한 세상에 대한 가정을 만족한다면,

> 관측의 오류는 없음. 일관적이지 않은 관측도 없음.

> 랜덤적인 요소가 없음.

> 시스템을 완전하게 설명할 수 있는 정보들을 다 수집함.
 
correct하게 수렴할 수 있다.

하지만 이 모든 것이 현실에서 이루어지기는 어렵다.

따라서 데이터의 사례의 개별 feature에는 noise가 끼는 것을 감안해야하고, 다른 결정요인이 있을 수도 있다. 

노이즈에 의해 맞는 가설이 제거 될 수 있다는 것이 현실의 문제이다. 그래서 맞다 아니다라고 말할 수가 없다.

# 2.3. Introduction to Decision Tree

어떻게 에러가 있는 데이터들을 기반으로 통계적인 기법을 가미해서 learning을 할 수 있을까?

가장 접근하기 쉬운 알고리즘 중 하나가 바로 Decision Tree.

## Decision Tree

예를 들어 <sunny,?,?,?,?,?>를 decison tree로 나타내면 아래와 같다.
![image](https://user-images.githubusercontent.com/101063108/157486541-987a78aa-51d8-47be-9f42-252a9b21b4ba.png)

<sunny,warm,?,strong,?,?>의 경우엔 다음과 같다.
![image](https://user-images.githubusercontent.com/101063108/157486810-86cad334-d620-4221-bc27-15c204f55249.png)

### Credit Approval Dataset

A1 ~ A15 까지 15개의 attribute가 있고, 총 690명의 사람 중 307명에게만 positive한 결과를 얻었다고 하자.
![image](https://user-images.githubusercontent.com/101063108/157488231-584c65f8-2638-4c97-b673-1eb6a6703fc2.png)

A1만을 가지고 판별하는 경우 
> a일 경우 98명이 positive하고 112명이 negative 

> b일 경우 206명이 positive하고 262명이 negative

> 데이터가 없을 경우 3명이 positive하고 9명이 negative

A9만을 가지고 판별하는 경우 
> t일 경우 284명이 positive하고 77명이 negative 

> f일 경우 23명이 positive하고 306명이 negative

어떤 attribute가 feature set에 포함되는 것이 좋을지 고민해 보아야한다.


# 2.4. Entropy and Information Gain

## Entropy

어떤 attribute를 더 잘 체크할 수 있을 것인지 알려주는 하나의 지표.

* 불확실성을 확실히 줄여주는 것이 좋다

이러한 불확실성을 재는 measure 중 하나가 entropy이다.

앞에서 봤던 A1이나 A9같은 attribute들을 특정 확률분포를 가지는 random variable로 생각할 수 있다. 이러한 random variable이 얼마만큼 불확실성이 높고 낮은지 판정하는 것이 entropy이다.

모든 케이스에 대해서 비슷한 확률 분포의 경우에는 불확실성이 높다 말할 수 있고, 어떤 하나의 케이스가 확실하게 항상 나온다면 불확실성이 없다고 말할 수 있다.

![image](https://user-images.githubusercontent.com/101063108/157492110-6c6e7f96-5259-4ed7-91c7-653e16f0ac56.png)

x : case 이산의 경우에는 합하면 되지만, 연속적이라면 적분을 하면 된다.

### conditional Entropy

어떤 특정 feature에 대한 정보가 있을 경우에 entropy를 판별하는 것.

![image](https://user-images.githubusercontent.com/101063108/157493822-e016724c-3c87-42dc-88a8-190600a06a5f.png)

주어진 x에 대해서 y의 불확실성을 측정하는 것. 따라서 밑에서의 P(X=x)의 summation 부분이 조건에 대해서 반영한 부분.

## Information Gain

앞의 A1과 A9 attribute에 대해 entropy를 계산해보자.

![image](https://user-images.githubusercontent.com/101063108/157510722-0fcded97-e7ca-4c97-9613-f09090cfb5e6.png)

y가 positive한 경우 (class variable) 확률은 다음과 같다.

P(Y=t) = 307/307+308

H(Y|A1)은 A1이라는 조건을 붙였을 때의 conditional entorpy값이고, H(Y|A9)는 A9라는 조건을 붙였을 때의 conditional entropy 값이다.

이때, Y라고 하는 class variable에 대해서 entropy가 주어졌을 때 어떤 attribute를 조건으로 줬을 경우 Y에 대한 entropy가 얼만큼 변했는지 그 차이를 보여주는 것이 Information Grain(IG)이다. 

𝐼𝐺(𝑌,𝐴𝑖)=𝐻(𝑌)−𝐻(𝑌|𝐴𝑖)

A1보다 A9이 IG가 높다.

### ID3 Algorithm

ID3, C4.5, CART등 다양한 decision tree가 있다. 그 중 하나인 ID3 Algorithm에 대해 알아보자.

![image](https://user-images.githubusercontent.com/101063108/157513974-3531f143-5312-482f-b07d-268afb5a22d4.png)


새로운 오픈 노드하나를 생성하고, instance들을 초기 노드에 넣는다.

가장 좋은 variable을 선택해 하위에 오픈 노드를 생성한다.
> 이때 어떤 attribute를 선택하면 좋을지는 IG를 활용하면 된다. 이런 식으로 하위에 오픈 노드들을 만들면 좀 더 큰 tree를 얻을 수 있다.

각 instance들을 선택된 variable에 맞게 분류하고 넣어준다. 

만일 정렬된 것들이 모두 동일한 class라면 그 노드를 닫는다.

### Problem of Decision Tree

현실은 '완벽한 세계'가 아니다. 

![image](https://user-images.githubusercontent.com/101063108/157515591-3e8d3be7-0388-49b4-b8b6-f7978c8d98ec.png)

실선 : 기존에 가지고 있던 데이터는 tree의 크기가 커짐에 따라 정확성도 커진다.

점선 : 새로 입력된 데이터는 tree의 크기가 커짐에 따라 정확성이 떨어진다. 세세한 판정들이 많아지고 에러에 대한 노이즈가 많아지면서, 정확성이 떨어진다.


# 2.5. How to create a decision tree given a training dataset

지금까지는 rule기반의 모델을 활용했다면 통계적인 방법에 대해서도 알아보자.

Housing dataset을 이용해볼텐데, 

* 13 numerical independent values (attribute)
* 1 numerical dependent value (class variable)

## Linear Regression

function approximation을 linear한 형태의 function으로 approximation하는 것

hypothesis를 linear한 함수 형태로 세우게 된다.

![image](https://user-images.githubusercontent.com/101063108/157518113-0c3630e0-8056-4d27-a785-33d0333d4a66.png)

n은 feature value ( independent value)의 개수

함수는 linearly weight sum 부분과 parameter 𝜽로 나뉘는데, 우리는 이 𝜃를 잘 정의하여 approximation을 진행한다.

더 나은 hypothesis를 위해 더 나은 𝜽를 찾아야한다.

![image](https://user-images.githubusercontent.com/101063108/157522466-af285720-c9cb-4efc-8e9c-4ee0f3240c4b.png)

x0를 1이라는 dummy variable을 붙여준다면, 𝜃0은 summation에 포함할 수 있다.

따라서 f hat = X𝜃 라는 값을 가지게 되고, 이는 행렬로 표시할 수 있다.

   행렬로 표시하면 X는 dummy variable 1을 포함하여 총 D개의 데이터셋과 n개의 attributes를 가진다고 할 수 있다. 

그러나 현실에서는 에러가 껴져있다. f hat과 f의 가장 큰 차이는 이 에러의 유무이다. f hat에 경우, 에러가 식에 포함되지 않는다. f가 현실에 맞는 true function이라고 할 수 있다. 식은 다음과 같다.

![image](https://user-images.githubusercontent.com/101063108/157523354-2f1553c5-112c-444f-9e4c-ef500cfa8a9d.png)

앞쪽의 summation 부분이 f hat과 일치하므로 f = X𝜃 + e = Y라고 할 수 있다.

𝜃는 에러가 가장 작을때로 해야한다. 따라서 다음과 같은 식이 전개된다.

![image](https://user-images.githubusercontent.com/101063108/157523755-13a1d32b-4587-427d-ad77-80a68072eac2.png)

𝜃 hat또한 마찬가지로 현실의 에러를 제외한 부분이다. 행렬의 원리를 활용해 식을 분해하고, 𝜃가 들어있지 않은 부분은 필요가 없으므로 식에서 제외한다.

이제 𝜃를 최적화해보자.

### Optimized 𝜃

위의 𝜃 hat의 식을 극값, 즉 미분을 활용한다.

![image](https://user-images.githubusercontent.com/101063108/157532292-388a5530-cbc9-4fb9-b860-b829e40c8423.png)

우리는 X, Y값을 알고있으므로 𝜃를 구할 수 있다.

housing data에 대해서 linear regression을 해보자. (첫번째 feature)

![image](https://user-images.githubusercontent.com/101063108/157533388-449b3cf1-bde2-43a8-abd0-e1c8eee6bc35.png)


Red : dependent variable의 분포 (f)

x라고 하는 하나의 값에 대해서 𝜃는 두개가 나온다. (dummy variable로 들어간것 하나: 𝜃0, feature에 대해서 하나: 𝜃1)

blue: 𝜃0에 의해 절편, 𝜃1에 의해 기울기가 생성되어 linear하게 만들어짐. (f hat)

뒤에 있는 부분까지 표현하기 위해서는 linear한 가정을 수정해야한다.

![image](https://user-images.githubusercontent.com/101063108/157534538-c30c6cf0-8d7e-4a04-a80d-312dc69f3efc.png)

![image](https://user-images.githubusercontent.com/101063108/157534939-fbfe4fc3-5175-473c-a63f-6a8b506712ba.png)

위의 linear regression이랑 같지만 x값을 x의 제곱, 세제곱,... 해서 수를 늘려주는 부분이 추가 된 것. (선형이 아님)

![image](https://user-images.githubusercontent.com/101063108/157535330-a8797ccc-dbb5-4c6e-a244-0f98af110396.png)

    red의 trend에 더 잘 맞음. 
    그러나 뒤의 부분이 관측치의 수가 적은데 이를 위해 승수를 높인다는 것은 노드를 늘리는 것 
    (트리 사이즈를 키우는 것)과 같다. 기존에 있는 것은 잘 설명 할 수 있지만, 미래에 올 값에 
    대해 잘 설명 할지는 의문.


### Limitations

* data set이 많아지면 많아질수록 모델이 복잡해지고 에러가 늘어날 것
* 한계가 있는 모델들이지만 기초가 되는 모델.




