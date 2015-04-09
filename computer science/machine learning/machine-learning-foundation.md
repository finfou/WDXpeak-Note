# 机器学习基石 学习笔记

Hsuan-Tien Lin htlin@csie.ntu.edu.tw

<!-- MarkdownTOC -->

- Roadmap
- Lecture 1 The Learning Problem
    - 从学习到机器学习
    - Machine Learning
        - 三个关键
    - 学习问题
- Lecture 2 Learn to Answer Yes/No
    - select g from H
    - Perceptron Learning Algorithm
    - Cyclic PLA
    - Linear Separability 线性可分
    - PLA Fact: W~t Gets More Aligned with W~f
    - PLA Fact: W~t Does Not Grow Too
    - 一个习题
    - More about PLA
    - Learning with **Noisy Data**
    - Pocket Algorithm
- Lecture 3: Types of Learning
    - Different Output Space
    - Different Label
    - Different Protocol
    - Different Input Space
- Lecture 4: Feasibility of Learning
    - Hoeffding's Inequality
    - Connect to Learning
- Lecture 5: Training versus Testing
    - Two Central Questions
    - Trade-off on M
    - Where Did M Come From
    - How Many Lines Are There?
    - Effective Number of Lines
    - Dichotomies: Mini-hypotheses
    - Growth Function
        - Growth Function for Positive Rays
        - Growth Fucntion for Positive Intervals
        - Growth Function for Convex Sets
    - The Four Growth Functions
        - Break Point of H
- Lecture 6: Theory of Generalization

<!-- /MarkdownTOC -->


## Roadmap

+ Lecture 1: The Learning Problem
    * _A_ takes _D_ and _H_ to get _g_
+ Lecture 2: Learning to Answer Yes/No
    * Perceptron Hypothesis Set
    * Perceptron Learning Algorithm, PLA
    * Guarantee of PLA
    * Non-Separable Data
+ Lecture 3: Types of Learning
    + Learning with Different Output Space y
        + [classification],[regression], structured
    + Learning with Different Data Lable y~n
        + [supervised], un/semi-supervised, reinforcement
    + Learning with Different Protocol f -> (x~n~,y~n~)
        + [batch], online, active
    + Learning with Different Input Space X
        + [concrete], raw, abstract
+ Lecture 4: Feasibility of Learning
    + Learning is Impossible?
    + Probability to the Rescue
    + Connection to Learning
    + Connection to Real Learning
+ Lecture 5: Training versus Testing
    + Recap and Preview
        + two questions: E~out~(g) ≈ E~in~(g), and E~in~(g) ≈ 0
    + Effective Number of Lines
        + at most 14 through the eye of 4 inputs
    + Effective Number of Hypotheses
        + at most m~H~(N) through the eye of N inputs
    + Break Point
        + when m~H~(N) becomes 'non-exponential'
+ Lecture 6: Theory of Generalization
    + Restriction of Break Point
    + Bounding Function: Basic Cases
    + Bounding Function: Inductive Cases
    + A Pictorial Proof

## Lecture 1 The Learning Problem

从基础学习 what every machine learning user should know

+ When Can Machines Learn? illustrative + technical
+ Why Can Machines Learn? theoretical + illustrative
+ How Can Machines Learn? technical + practical
+ How Can Machines Learn Better? practical + theorietical

知其然也知其所以然

### 从学习到机器学习

+ 学习的过程：observations -> **learning** -> skill
+ 机器学习的过程：data -> **ML** -> skill(improved performance measure)
+ skill: improve some **performance measure**

### Machine Learning

+ improving some perormance mearsure with experience computed from data
+ an alternative route to build complicated systems

#### 三个关键

+ exists some 'underlying pattern' to be learned, so performance measure can be improved. 要有东西可学
+ but no programmable definition, so 'ML' is needed
+ somehow there is data about the pattern. 要有大量数据


### 学习问题

+ 输入 x
+ 输出 y
+ 目标函数 target function f: X->Y
+ data D={(x~1~,y~1~),(x~2~,y~2~),...,(x~n~,y~n~)}
+ 机器学习可能得到的假设 g:X->Y
+ {(x~n~, y~n~)} from f -> ML -> g

 ![learning flow](./_resources/mlf1.jpg)

+ f 我们不知道
+ g 越接近 f 越好
+ A takes D and H to get g
+ related to DM, AI and Stats

## Lecture 2 Learn to Answer Yes/No

+ 每一个样本的数据可以看成一个向量，可以给每一个向量计算出一个加权得分，每一个维度有一个权重。
+ 把 threshold 收进公式中，可以得到一个统一的表达，最后的得分等于两个向量相乘
+ ![mlf2](./_resources/mlf2.jpg)
+ perceptrons <-> linear(binary) classifiers 线性分类器

### select g from H

* H = all possible perceptrons, g = ? 从这么多可能的线之中，选出一条最好的，最能区分数据的
* 先要求 g 和 f 在已有数据上结果最接近, g(x~n~) = f(x~n~) = y~n~
* 难点在于，H 很大，有无数种可能的线(分类器)
* 从第一条线 g~0 开始，不断进行修正，可以认为是一开始的权重向量 w~0

### Perceptron Learning Algorithm

* For t = 0, 1, ... 这里 t 是轮数，因为会迭代很多次
* 找到 w~t 的一个分类错误的点(x~n(t)~, y~n(t)~), 即 sign(w~t~^T^x~n(t)~) 不等于 y~n(t)~
* 试着去改正这个错误 w~t+1~ <- w~t + y~n(t)~x~n(t)~ until no more mistakes
* 返回最后得到的 w 为 g, 这个 w 称为 w~pla~

### Cyclic PLA

* For t = 0,1,...
* find the next mistake of wt called (x~n(t)~, y~n(t)~), aka sign(w~t~^T^x~n(t)~) 不等于 y~n(t)~
* correct the mistake by w~t+1~ <- w~t + y~n(t)~x~n(t)~
* until a full cycle of not encountering mistakes
* 可以采用标准的遍历，或者也可以是预先计算好的随机顺序

### Linear Separability 线性可分

* if PLA halts, (necessary condition) D allows some w(一条用来区分的线) to make no mistake
* 有一条线可以区分数据，即有解，有解的时候 PLA 算法才会停

### PLA Fact: W~t Gets More Aligned with W~f

+ 线性可分，则存在一条完美的直线 W~f(即目标函数) 使得 y~n = sign(W~f^T x~n)
+ 也就是 y~n 的符号，与 W~f^T 和 x~n 的乘积(也就是 x~n 到直线 W~f 的距离)的符号，一定是相同的
+ ![mlf3](./_resources/mlf3.jpg)
+ W~t 为当前次迭代的直线，找出一个错误的点，然后做更新
+ 通过不等式可以得到，下一次迭代得到的直线，会更加接近于完美的直线 W~f ，因为乘积越来越大了(但是乘积还需要考虑向量的长度，这里说的是角度，下面就是说长度)

### PLA Fact: W~t Does Not Grow Too

+ W~t changed only when mistake
+ 也就是只有在 sign(W~t^T x~n(t)~) 不等于 y~n(t)~ 也就是 y~n(t)~w~t^T^x~n(t)~ <= 0
+ ![mlf4](./_resources/mlf4.jpg)
+ 平方之后来看长度的公式，蓝色部分通过上面的推导可知是小于等于零的
+ y~n 是正负 1，所以下一次迭代的向量的长度的增长是有限的，最多增长 x~n~^2^ 那么多(也就是长度最大的向量)
+ 然后推导出来 w~t 确实是越来越靠近 w~f 的

### 一个习题

+ ![mlf5](./_resources/mlf5.jpg)
+ 具体怎么推导的呢，研究了半个多小时终于弄清楚了，如下
+ W~f 是理论上完美的那条线，w~t 是第 t 次迭代得到的那条线，而因为这两条线最好结果就是完全平行，所以有
    + (W~f^T / || W~f ||) * (w~t / || w~t ||) 的最大值为 1 (`eq1`)
+ 由前面 PPT 得到的两个公式：
    + W~f~^T^w~t+1~ >= W~f~^T^w~t~ + min~n~y~n~W~f~^T^x~n~ (`eq2`)
    + w~t+1~^2^ <= w~t~^2^ + max(n)x~n~^2^ (`eq3`)
+ 因为是迭代 T 次，把 `eq2` 和 `eq3` 代入到 `eq1` 中
    + W~f~^T^w~t~ / || w~f~ || 这部分就是条件里的 p，因为迭代 T 次，所以分子变成 T·p
    + 分子是 || w~t ||，根据 `eq3` 可知迭代 T 次后为 √(T·R^2^)
+ 又因为 `eq1` 的最大值为1，可以求出 T 的范围，得到答案

### More about PLA

+ Guarantee: as long as **linear separable** and **correct by mistake**
    + inner product of w~f and w~t grows fast; length of w~t grows slowly
    + PLA 'lines' are more and more align with W~f -> halts
+ Pros
    + Simple to implement, fast, works in any dimensin d
+ Cons
    + **'Assumes' linear separable D** to halt(property unknown in advance)
    + Not fully sure **how long halting takes**(p depends on W~f) -though practically fast
+ What if D not linear separable?

### Learning with **Noisy Data**

+ ![mlf6](./_resources/mlf6.jpg)
+ Line with Noise Tolerance
    + ![mlf7](./_resources/mlf7.jpg)
    + 在看到的数据中，找犯错误最少的一条(但是这是一个很难的问题)

### Pocket Algorithm

+ modify PLA algorithm (black lines) by **keeping best weights in pocket**
+ ![mlf8](./_resources/mlf8.jpg)
+ 例题时间
+ ![mlf9](./_resources/mlf9.jpg)

## Lecture 3: Types of Learning

### Different Output Space

+ 二元分类与多元分类
+ 例子：Patient Recovery Prediction Problem
    + binary classification: patient features -> sick or not
    + multiclass classification: patient features -> which type of cancer
    + regression(回归分析)
        + patient features -> how many days before recovery
        + compnay data -> stock price
        + climate data -> temperature
+ 统计上有很多工具也可以放到机器学习里来用
+ Structured Learning
    + Sequence Tagging Problem 词性标注
    + protein data -> protein folding
    + speech data -> speech parse tree

### Different Label

+ 监督式学习
    + ![mlf10](./_resources/mlf10.jpg)
+ 非监督式学习
    + ![mlf11](./_resources/mlf11.jpg)
+ 其他一些非监督式学习
    + ![mlf12](./_resources/mlf12.jpg)
    + diverse, with possibly very different performance goals
+ 半监督式学习
    + 只提供有限信息，只标记一部分，蓝色的是没有标记的，其他颜色是标记出来的
    + ![mlf13](./_resources/mlf13.jpg)
    + leverage unlabeled data to avoid 'expensive' labeling

### Different Protocol

+ Reinforcement Learning
    + 惩罚错误判断，鼓励正确判断
    + learn with **partial/implicit information**(often sequentially)
+ Batch Learning 填鸭式教育
    + ![mlf14](./_resources/mlf14.jpg)
    + batch learning: **a very common protocol**
+ Online 老师教书
    + hypothesis 'improves' through receiving data instances sequentially
    + PLA can be easily adapted ton online protocol
    + reinforcement learning is often done online
+ Active Learning
    + ![mlf15](./_resources/mlf15.jpg)

### Different Input Space

+ Concrete features: the 'easy' ones for ML
+ Raw features -> meaning of digit(比方说笔迹识别，把图像转换成数字化的信息，可以是对称性或者密度，或者直接转化成二维数组，越抽象，对于机器来说就越困难)
    + often need human or machines to **convert to concrete ones**
    + 要么是人工来做，要么就是 deep learning 自动来做
+ Abstract features: **no** physical meaning, even harder for ML
    + 比方说评分预测问题(KDDCup 2011)
    + need **feature conversion**/extraction/construction

## Lecture 4: Feasibility of Learning

+ Inferring Something Unknown
    + diificult to infer **unknown target f outside D** in learning
    + 抽样调查
+ **Hoeffding's Inequality**
    + 只是给出一个比较高的上限
    + probably approximately correct(PAC)

### Hoeffding's Inequality

用一个从罐子里取玻璃球作为例子，有两种玻璃球：橙色和绿色。假设：

    橙色的概率为 u
    则绿色的概率为 1-u
    但 u 具体是多少我们不知道

然后我们从中取出 `N` 个样本：

    橙色的比例为 v
    则绿色的比例为 1-v
    这时我们是知道 v 具体是多少的

> Does **in-sample v** say anything about out-of-sample u?

+ Possibly not: sample can be mostly green while bin is mostly orange
+ Probably yes: in-sample v likely **close to** unknown u

> Formally, what does v say about u?

    u = orange probability in bin
    v = orange fraction in sample

![mlf16](./_resources/mlf16.jpg)

抽样数量足够大的时候，抽样得到的概率 v 和实际概率 u 相差的概率会很小(Heoffding's Inequality)

The statement `v = u` is **probably approximately correct**(PAC)

根据上面的公式我们知道，其实要知道抽样得到的概率 v 和实际概率 u 相差多少，只跟误差和抽样的数量有关。

If **large N**, can **probably** infer unknown u by known v

### Connect to Learning

瓶中的情况 | 对应到 Learning
--- | ---
unknown orange prob. u | fixed hypothesis h(x) ? target f(x)
marble in bin | x 在 X 中
orange | h is wrong -> h(x) 不等于 f(x) aka orange
green | h is right -> h(x) 等于 f(x) aka green
size-N sample from bin | check h on D = {(x~n~, y~n~)} 这里 y~n~ 就是 f(x~n~)

![mlf17](./_resources/mlf17.jpg)

![mlf18](./_resources/mlf18.jpg)

E~out~(h) 对应于 总体概率 u，E~in~(h) 对应于抽样概率 v

![mlf19](./_resources/mlf19.jpg)

Does not depend on E~out~(h),**no need to know E~out~(h)**

E~in~(h) = E~out~(h) is **probably approximately correct**(PAC)

![mlf20](./_resources/mlf20.jpg)

**BAD** sample: **E~in~ and E~out~ far away**

can get **worse** when involving 'choice'

Hoeffding 保证的是出现 Bad Sample 的机会不会很大

![mlf21](./_resources/mlf21.jpg)

什么意思呢？如果hypothesis set是有有限种选择，训练样本够多，那么不管学习算法A怎么选择，样本的判别结果都会与总体的一致。那么，如果学习算法设计为寻找样本中错误率最小的，那么刚刚的推论PAC就能保证选出来的g与f是约等于的。

也就是说，当有 M 个 hypothesis 的时候，对应的误差也会变大，但是依然可以找到一个情况，此时

E~in~(g) = E~out~(g) is **PAC, regardless of A**

Most reasonable A(like PLA/pocket): pick the h~m with **lowest E~in~(h~m~) as g

![mlf22](./_resources/mlf22.jpg)

不过仍然有一个遗留问题，刚刚的推论是在hypothesis set有限的前提下，那类似于PLA的hypothesis set是无穷的又如何呢？不用紧张，以后会证明这个问题。现在至少在有限的情形下证明了，这是一个很好的出发点。

## Lecture 5: Training versus Testing

![mlf23](./_resources/mlf23.jpg)

### Two Central Questions

![mlf24](./_resources/mlf24.jpg)

+ Can we make sure that E~out~(g) is close enough to E~in~(g)
+ Can we make E~in~(g) small enough?

### Trade-off on M

M 也就是 hypothesis 的集合，对于不同的值，对于上面两个问题有两个不同的解答

Small M | Large M
--- | ---
Yes! P[BAD] <= 2·M·exp(...) | No! P[BAD] <= 2·M·exp(...)
No! too few choices | Yes!, many choices

这样就两难了，M 小的时候，可以保证 E~out~(g) 和 E~in~(g) 足够接近，但是不能保证 E~in~(g) 足够小；M 大的时候则是相反的情况。如何选择正确的 M 呢？尤其是 M 可能是无限多的情况怎么办呢？

现在的情况就是这个公式和 M 有关

![mlf25](./_resources/mlf25.jpg)

所以想办法能不能把可能无限大的 M，弄成有限的数量 m~H

![mlf26](./_resources/mlf26.jpg)

### Where Did M Come From

![mlf27](./_resources/mlf27.jpg)

之前的讨论中，我们直接把概率的 or 转化为概率的相加，因为我们假设这些 BAD case 不大可能会重叠，但是当 M 很大的时候，这种做法(Uniform Bound)就不行了。为什么呢？

因为假如 h~1 很接近于 h~2~,则 E~out~(h~1~) 会很接近 E~out~(h~2~)，并且很有可能 E~in~(h~1~) = E~in~(h~2~)

Union bound **over-estimating** 很多重复的也被计算进去了。那么对于这种情况，如果我们能 group similar hypotheses by **kind**，就可以减少误差。

### How Many Lines Are There?

考虑平面上所有的线 H = {all lines in R^2^}

+ How many lines? 无限多条
+ How many **kinds of** lines if viewd from one input vector x~1?
    + 两种，一种划分 x~1 是圈，另一种划分 x~1 是叉
    + 2 kinds: h~1~-like(x~1~) = o or h~2~-like(x~1~) = x

如果从两个点的角度来看呢？有 4 种线，x~1 和 x~2 的组合可能是：oo, xx, xo, ox

三个点的情况下呢？通常情况下有 8 种线，x~1~, x~2~, x~3~ 的组合可能是：ooo, xxx, oox, xxo, oxo, xox, oxx, xoo

但是如果三点共线的话，就只有 6 种线了。

四个点的情况下呢？至多只有 14 种线了。(指的是线性可分的线的种类的数量)

### Effective Number of Lines

maximum kinds of lines with respect to N inputs x~1~, x~2~, ..., x~N~ -> **effective number of lines** 可能的有效划分的线种类数量

![mlf28](./_resources/mlf28.jpg)

这里用 `effective(N)` 代替了原来的 `M`，如果 `effective(N)` 远小于 2^N 的话，那么就可能解决无穷条线的问题了

### Dichotomies: Mini-hypotheses

![mlf29](./_resources/mlf29.jpg)

用 Dichotomies Set 的大小来代替 M，但是现在会依赖于不同点的选取，需要做进一步的修改

### Growth Function

![mlf30](./_resources/mlf30.jpg)

注意这里这个 m~H~(N) 是重要的符号

#### Growth Function for Positive Rays

![mlf31](./_resources/mlf31.jpg)

N+1

#### Growth Fucntion for Positive Intervals

![mlf32](./_resources/mlf32.jpg)

0.5N^2 + 0.5N + 1

#### Growth Function for Convex Sets

![mlf33](./_resources/mlf33.jpg)

![mlf34](./_resources/mlf34.jpg)

### The Four Growth Functions

![mlf35](./_resources/mlf35.jpg)

#### Break Point of H

指的是增长函数中第一个有希望的点(也就是增长趋势放缓的点)，比方说之前三个点我们可以做出 8 种线，但是四个点的时候却不能做出 16 种线，所以 4 就是一个 break point。

![mlf36](./_resources/mlf36.jpg)

## Lecture 6: Theory of Generalization


