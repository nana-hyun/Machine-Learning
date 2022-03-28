# 5.1. Decision Boundary with Margin

앞서서 Decision boundary를 결정하는 여러 확률론적 방법들에 대해 배웠었는데,

첫번째로, Bayes Risk를 활용한 방법이 있었다.

![image](https://user-images.githubusercontent.com/101063108/160315868-171f21c6-7b26-465a-9ce7-a86a422bce3c.png)

빨간 선과 초록 선이 교차하는 부분이 decision boundary가 된다.

두번째로, 2차원에서의 decision boundary를 결정하는 것이 있는데, 이는 Naive Bayses를 이용하여 classify해 주었다.

![image](https://user-images.githubusercontent.com/101063108/160316062-d7dcb545-f78c-4dc3-9bc2-727dafa949d4.png)

위의 두 경우 모두 **확률**에 의존한 decision boundary의 결정이다.

그렇다면 확률 없이 decision boundary를 찾을 수 있을까?

![image](https://user-images.githubusercontent.com/101063108/160316233-100b2395-320d-4e96-952d-892960f3beb2.png)

이런 instance들이 있다고 가정해보자. 

이 경우, decision boundary를 아래와 같이 설정했을 때 어떤게 best한가?

![image](https://user-images.githubusercontent.com/101063108/160316432-c8ead086-8705-4219-8cb1-89af12d7275d.png)

연두색 선과 주황색 선의 경우 선에 근접한 case들이 오류를 범할 확률이 증가한다. 

따라서 하늘색 선이 가장 적절하다고 볼 수 있다.

이러한 거리를 최대로 할 수 있는 경우, decision boundary를 설정할 수 있다.

![image](https://user-images.githubusercontent.com/101063108/160316923-7b000003-c852-42bb-86d6-dc09cba2c84e.png)

위 그림에서 보면, 경계에 가장 가까운 빨간 점 두개를 지나는 직선을 그리고, 그 직선을 평행하게 경계에 가장 가까운 파란 점까지 내려주어, 한계치를 설정해준다.

그 후 그 두 직선의 중간 지점에 평행한 직선을 그리면, 그것이 decision boundary가 된다.

이때, decision boundary와 한계에 위치한 직선사이의 거리를 Margin이라고 한다.

(정확하게 이야기하자면, decision boundary와 가장 가까운 점 사이의 거리이다.)

여기서 이 decision boundary를 결정짓는 point 3개를 찾는 것이 핵심이 된다.

*support vector machine은 이러한 점(벡터)들이 decision boundary를 결정하기 위해 support해준다는 의미*

decision boundary line은 아래와 같이 정의할 수 있다.

* **w · x** + b = 0

w는 x1과 x2라는 두개의 매개변수를 가지는 벡터로 표현, b는 절편

-> 총 3개의 매개변수가 필요하다.

* positive case
    * **w · x** + b > 0
* negative case
    * **w · x** + b < 0
* confidence level
    * (**w ·** 𝐱𝒋 + b)𝑦𝑗

이때 confidence level을 최대화하는 것이 적절한 decision boundary를 설정하는 방법이다.

# 5.2. Maximizing the Margin

f(x) = **w · x** + b 라고 하자.

![image](https://user-images.githubusercontent.com/101063108/160327833-aa86dc37-f291-490e-99f3-68dbac3e843c.png)

점 x가 boundary 위에 있다면, f(x) = **w · x** + b = 0

양의 점 x라면, f(x) = **w · x** + b = a, a>0

margin distance를 재보자.

