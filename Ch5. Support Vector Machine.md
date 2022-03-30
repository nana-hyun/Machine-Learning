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

임의의 점 x에서 decision boundary 위로 수선의 발을 내린 점을 ![image](https://user-images.githubusercontent.com/101063108/160341634-af05dedb-3103-4fe5-8332-fc52c5f30efb.png)라 하자.

그 둘 사이의 거리를 재볼텐데, 이 거리를 r이라 하고, w는 decision boundary에 수직인 벡터 (1,-1)이라 하면,

![image](https://user-images.githubusercontent.com/101063108/160345653-7cf5d1af-00f7-42c5-ba36-f8a917a1c3c3.png)

w벡터를 w벡터의 크기로 나눠주면 단위벡터가 된다.

![image](https://user-images.githubusercontent.com/101063108/160345779-2f6c6053-7550-4700-9205-7728edeb3316.png)

이때 ![image](https://user-images.githubusercontent.com/101063108/160345887-7ef0cbb8-d80b-43c9-8425-74eb0b0a2a23.png)는 위의 식에서 0이므로, 위와 같이 계산된다.

따라서 거리 r은 다음과 같다.

![image](https://user-images.githubusercontent.com/101063108/160345995-e3554d7d-f7a9-4a77-ad68-9dfad8420748.png)

그렇다면 좋은 decision boundary는 margin distance를 어떻게 해야할까?

margin을 최대화해야 한다.

여기서 점과 decision boundary 사이의 거리가 r, 즉 위의 식과 같다.

![image](https://user-images.githubusercontent.com/101063108/160351175-aa0e8a97-8e3d-487f-93ef-31159656adf6.png)

이때, 아래쪽 한계와 위쪽 한계 둘다 고려해주어야 하므로 2r을 최대화 하고자 한다.

![image](https://user-images.githubusercontent.com/101063108/160351361-93ad5649-ef8a-4399-b318-264ef2b89b01.png)

우리는 margin distance를 구하고 있으므로, 모든 instance에 대해 a보다 크거나 같아야한다.

a는 임의의 숫자 이므로 1을 넣어보면,

![image](https://user-images.githubusercontent.com/101063108/160352301-3b2f648f-b018-480e-a88f-0833df7a9912.png)

위의 식처럼 되고, 2는 상수이므로 최대화하는데 영향을 미치지 않으므로 1로 바꿔도 무방하다.

w가 분모에 있어 계산하기 까다로우므로, 최소화하는 것으로 식을 바꿔준다.

![image](https://user-images.githubusercontent.com/101063108/160352676-acebb40a-2d10-4156-9d3a-08bbdc6e3be0.png)

이때 ![image](https://user-images.githubusercontent.com/101063108/160353160-9a82e026-d077-4b9b-853e-1325610c886e.png)
는 ![image](https://user-images.githubusercontent.com/101063108/160353184-f717d8c6-d82a-4370-a7fd-0462d587e8f2.png) 으로 쓸 수 있고, 루트의 경우 연속적으로 증가하기 때문에 최소화하는데에 영향을 미치지 않는다.

따라서 안의 내용이 중요해지는데 이 제곱의 형태가 있어서 이것이 quadratic optimization 문제가 된다.

linear programming을 이용하거나, quadratic programming을 이용해 최적화를 수행할 수 있다.

# 5.3. SVM with Matlab

//matlab 실습

우리가 지금까지 해왔던 것은 분류가 잘 되어 있는 경우 였다. 하지만 실제로는 그렇지 않은 경우가 많을 것이다.

![image](https://user-images.githubusercontent.com/101063108/160525663-163b30eb-7639-441d-8a3b-ebf7dfe1f1d3.png)

만일 새로운 점들이 추가 된다고 하면, Decision boundary도 변경될 것이다.

![image](https://user-images.githubusercontent.com/101063108/160525740-bdd5104d-a3e4-4265-b62f-914da005e4c9.png)

위의 두개의 점이라면, margin이 줄어들긴 해도 선형으로 decision boundary를 결정할 수 있을 것이다.

그러나 분류를 어렵게 하는 점이 들어온다면, 선형으로는 분류를 할 수 없는 경우가 생긴다.

우리는 그동안 Hard margin을 이용해왔다.

Hard margin이란, 에러 케이스를 허용하지 않는 것이다. 반대로 Soft margin은 지정한 에러케이스는 허용하겠다 라는 의미이다.

이러한 soft margin을 이용해 이 문제를 해결할 수 있고, 또는 kernel trick을 이용해 hard margin이지만 decision boundary가 선형이 아닌 방식으로 해결할 수 있을 것이다.

# 5.4. Error Handling in SVM

이러한 선형적인 decision boundary로 분류할 수 없는 에러 케이스가 발생했을 때 어떻게 해야할까?

![image](https://user-images.githubusercontent.com/101063108/160537164-0d60b885-b37c-4167-9d92-4e007b408de0.png)


1. decision boundary를 더 복잡하게 만들자.

그렇게 되면 이제 non-linear하게 된다.

2. error가 있음을 인정하자.

이러한 에러를 식에 적용하고, 이 에러를 잘 줄이는 것이 중요해진다.

에러가 있음을 인정할 때, 어떻게 다루어야 할까?

![image](https://user-images.githubusercontent.com/101063108/160537902-04e11df5-ead0-4085-ac81-c21e5c324e75.png)


1. 에러 케이스의 개수를 세서, 그 수를 줄이는 방법

![image](https://user-images.githubusercontent.com/101063108/160537274-a26fb113-b26e-4ef9-85e2-a9c5964bc661.png)

C는 정도를 나타내는 상수, 1이라고 둘 수 있다. C를 1이라 두면, 에러 케이스들은 멀든 가깝든 모두 1이라는 값을 가지게 된다.

파란색 점 부분이 positive하다고 가정하고, 파란 점 하나가 점점 decision boundary 쪽으로 이동하다가 넘어가서 error case가 된다고 해보자.

이 점이 파란 선에 닿는 순간 우리가 전에 말한 a라는 값이 1이 되고, 거기서부터 에러의 가능성은 올라가게 된다.

그러다가 decision boundary를 넘는 순간 이 케이스는 이제 에러가 되므로 에러를 나타내는 loss function은 0에서 1로 점프하게 된다.

이러한 방식은 현재는 잘 사용되지 않는데, 멀거나 가깝거나 상관없이 error 케이스가 다 1이라는 penalty를 받게 되기 때문이기도 하다.

2. Hinge loss의 이용

이러한 것을 보완한것이 hinge loss이다.

hinge loss는 slack 변수를 사용하여 잘못 분류된 case에 대해 각기 다른 penalty를 부과한다.

![image](https://user-images.githubusercontent.com/101063108/160538157-db955979-71a8-4459-b847-1fd05620f84e.png)

![image](https://user-images.githubusercontent.com/101063108/160538175-1cd9a244-9251-4ccc-bf8c-c74c118a18b5.png)

식에서는 모든 slack변수를 더한 값이 들어가게 된다.

여기서 C는 상수가 아닌, 정도가 다른 매개변수가 된다.

a = 1이 되는 시점부터 서서히 증가하기 시작하며, decision boundary를 지나는 순간 1이 되고 그 후는 그 이상이 된다.

조건 또한 뒤에 ![image](https://user-images.githubusercontent.com/101063108/160538606-5e54fef4-7615-4fe0-9c3f-c029df749402.png) 이것 만큼이 되는 slack이 붙을 수 있다는 것으로 바뀐다.

여기서의 문제는 C가  매개변수이기 때문에 C의 값을 정해줘야 한다는 것이다.

# 5.5. Soft Margin with SVM

![image](https://user-images.githubusercontent.com/101063108/160541790-7f4962d2-bcb4-426a-b9b3-5bde6bee92a7.png)

slack 변수를 추가하여 penalty를 줄 수 있는데, decision boundary를 넘어가면 penalty가 1 이상의 값을 가지게 된다. 

penalize를 하는 function은 다음과 같이 slack을 총 합하는 것으로 나타낼 수 있다.

![image](https://user-images.githubusercontent.com/101063108/160542258-72c4dfd4-e7e4-4d14-a900-6ce7905e3e3a.png)

penalty의 정도를 결정하는 이 C를 정하는게 중요해진다.

**Log Loss**

![image](https://user-images.githubusercontent.com/101063108/160544788-78ce3e5d-d60e-4eba-87b0-16e21f2d7c8f.png)

logistic regression을 이용해 loss function을 만들 수 있는데, 이를 log loss라고 한다.

![image](https://user-images.githubusercontent.com/101063108/160544988-200eeee1-0c16-4c0c-b1b5-bab05599ec4c.png)

![image](https://user-images.githubusercontent.com/101063108/160545024-2d0bdea8-3d9c-499e-8df9-60786770ce4f.png)

예전에 했던 식을 적용시키면,  hinge loss의 ![image](https://user-images.githubusercontent.com/101063108/160545181-02a2438a-9bc9-4083-97f3-90e26a0c29de.png) 식과 구조가 비슷함을 알 수 있다.

세개의 loss function (zero-one loss, hinge loss, log loss) 중 어떤 것이 가장 선호될까?

hinge loss가 1부터 penalty가 증가하는 것으로 보이는 것에 반해 log loss를 보면 이미 penalty값을 가지고 있는 것을 확인할 수 있다. decision boundary를 지난 후부터의 penalty를 보면, hinge loss에 비해 완만하게 증가하는 것을 볼 수 있다.

그렇다면 C는 어떻게 정하는게 좋을까?

![image](https://user-images.githubusercontent.com/101063108/160546051-18db529e-5a77-4f2e-a5a3-df7c35938833.png)

아래 그림을 보면 알 수 있듯이, C의 크기가 너무 작을 경우에는 penalty가 너무 작아지므로 제대로된 decision boundary를 설정할 수 없다. 일정 크기이상의 C는 decision boundary의 위치가 거의 변하지 않는 것을 확인할 수 있다.

# 5.6. Rethinking of SVM

![image](https://user-images.githubusercontent.com/101063108/160550511-09279757-daa4-4ff1-9342-d56bf0d77aa6.png)

지금까지는 error case를 인정하고, 최소한의 에러를 만드는 soft margin에 대해 다뤘다면, 이제는 더 복잡한 decision boundary를 통해 error case를 만들지 않는 방법을 이용하고자 한다.

이러한 복잡한 decision boundary를 설정하기 전에, 가지고 있는 데이터들이 어떤 방법에 적합한지 먼저 판단하는 과정이 필요하다. 

![image](https://user-images.githubusercontent.com/101063108/160551493-d636e9b4-2394-478f-91d7-ba0904a91d31.png)

linear한 decision boundary를 결정하기 위한 그림이다. 이런 사례에서는 명확하게 분류가 되는 hard margin이지만, 아래의 그림을 보자.

![image](https://user-images.githubusercontent.com/101063108/160551736-78136915-13b4-4a27-b6ab-76f3187858ad.png)

이러한 경우에서는 위와 같은 방법을 사용할 수 없다. 더 복잡한, non-linear한 decision boundary가 필요하다.

따라서 우리는 전에 linear regression에서 차수를 높여서 계산했던 것처럼, 주어진 x1, x2를 조합해 차원를 높여 넣어볼 것이다.

![image](https://user-images.githubusercontent.com/101063108/160552581-0d6ce1c0-36f9-49b8-b153-3a0d2c3d191a.png)

그렇게 하면 오른쪽과 같은 그림이 나오고, 연두색 부분을 따라 decision boundary가 형성된 것을 볼 수 있다.

여기서는 3차까지 높여서 넣어보았는데, 이것을 무한대까지 늘리기 위해서는 kernel trick을 사용해야한다.

kernel trick을 사용하기 위해서는 SVM에 있는 primal 문제를 dual 문제로 바꾸어야한다.

SVM은 Constrained quadratic programming을 이용해서 parameter를 추론했다.

Constrained optimization 

![image](https://user-images.githubusercontent.com/101063108/160554390-81756d97-4c97-40fd-b758-b78429295887.png)


**Lagrange method**

![image](https://user-images.githubusercontent.com/101063108/160554500-0cd0c849-72c8-4fb4-b1ac-cfe8557b4b6c.png)

optimize 되어있는 Lagrange의 prime function은 f(x)와 동일하게 작동한다.

그렇게 되면, f(x)를 쓰지 않고 위의 함수를 사용할 수 있다 : primal problem -> dual problem

# 5.7. Primal and Dual with KKT Condition

![image](https://user-images.githubusercontent.com/101063108/160765925-3f946193-bd07-48eb-b1c6-4f7d987bf6bd.png)

primal problem과 lagrange dual problem은 위와 같으며, 앞에서의 lagrange method와 관련하여 설명될 수 있다.

**strong duality**

dual problem의 solution과 primal problem의 solution이 같아야 한다.

![image](https://user-images.githubusercontent.com/101063108/160766592-e7a9f00c-78f2-4377-87ed-6b390ee24e2e.png)

이러한 strong duality를 보장해주는 것이 있는데 Karush-Kunh-Tucker (KKT) 조건을 만족시키는 것이다.

여기서는 dual problem으로 넘어갈 때, KKT condition이 만족된다고 감안하고 넘어가자.

진짜 이 파트 뭔 말인지 이해안되니까 나중에 다시 정리..

# 5.8. Kernel

**Mapping Function**

![image](https://user-images.githubusercontent.com/101063108/160775334-9a4b76e0-1da2-4f48-98b4-bce25d081958.png)

2차원 위의 공간에서, 각 점들을 잇는 선분들이 분리되어 있지 않은 것을 확인 할 수 있다.

이 점들을 3개의 변수?로 만들어주면, 우리는 3차원 공간에서 이 점들을 분리해서 표현할 수 있다.

이러한 원리를 이용해서 3차, 4차,... 100차 혹은 그 이상도 가능할 것이다. 이런 특성 공간이 계속 커진다는 단점이 있다.

kernel 계산은 다른 공간(차원)에서 두개의 벡터를 내적하는 것이다.

![image](https://user-images.githubusercontent.com/101063108/160776725-d1a5810c-151f-4d54-8d7e-ab06835793aa.png)

* polynomial(homogeneous)
* polynomial(inhomogeneous)
* Gaussian kernel function
* Hyperbolic tangent

polynomial kernel function의 경우, 다음을 따른다.

![image](https://user-images.githubusercontent.com/101063108/160777165-51e7bebc-a9c4-46c5-a35b-c1a008ad0834.png)

원본 벡터들을 다른 차원으로 옮긴 뒤, 내적하는 것과 내적을 한 뒤 다른 차원의 계수만큼 제곱해주는 것은 같은 결과임을 알 수 있다.

이를 이용하면 매우 높은 차원으로 옮기고 내적하는 kernel을 내적하고 제곱 계산을 해주는 것으로 간단히 해결할 수 있다.

# 5.9. SVM with Kernel


