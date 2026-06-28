# LLM

# agent



## 定义

智能体被定义为任何能够通过**传感器（Sensors）**感知其所处**环境（Environment）**，并**自主**地通过**执行器（Actuators）**采取**行动（Action）**以达成特定目标的实体。



传统视角下的智能体经历了一条从简单到复杂、从被动反应到主动学习的清晰演进路线

+ **反射智能体**：结构最简单，它们的决策核心由工程师明确设计的“条件-动作”规则构成，自动恒温器是其中一种
+ **基于模型的反射智能体**：内部拥有**世界模型（智能体对外部世界运行规律的内部表示）**，用于追踪和理解环境中那些无法被直接感知的方面
+ **基于目标的智能体**：主动地、有预见性地选择能够导向某个特定未来状态的行动
+ **基于效用的智能体**：最大化期望效用，在相互冲突的目标之间进行权衡
+ **学习型智能体**：不依赖预设，而是通过与环境的互动**自主学习**（强化学习）进行决策



由大语言模型驱动的 LLM 智能体，其核心决策机制与传统智能体存在本质区别；**传统智能体的能力源于工程师的显式编程与知识构建，其行为模式是确定且有边界的；而 LLM 智能体则通过在海量数据上的预训练，获得了隐式的世界模型与强大的涌现能力，使其能够以更灵活、更通用的方式应对复杂任务。**

![图片描述](assets/1757242319667-2.png)

因此，LLM 智能体可以直接处理**高层级、模糊且充满上下文信息**的自然语言指令



智能体的任务环境通常使用PEAS描述，以智能旅行助手为例	

![image-20260620190132443](assets/image-20260620190132443.png)













# RL

强化学习的目标是**求解最优策略**



## basic concept

**State**：环境给agent的信息

**Action**：agent在某个状态下可以执行的决策

**Policy**: agent在某个状态下选择动作的原则，$$\pi\left(a_{2}\mid s_{1}\right)=1$$表示在s1的状态下采取动作a2的概率为1	

**reward**：$$p\left(r=-1 \mid s_{1}, a_{1}\right)=1$$在状态s1和a1下得到分数-1的概率为1

**Trajectory**：state-action-reward链 $$ S_1 \xrightarrow[\,r=0\,]{a_2} S_2 \xrightarrow[\,r=0\,]{a_3} S_5 \xrightarrow[\,r=0\,]{a_3} S_8 \xrightarrow[\,r=1\,]{a_2} S_9. $$ 该链上的reward之和为return，return可用于评估policy；存在Terminal state的Trajectory称为eposide

**Discounted reward**：$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$$，用于解决无限发散的序列，γ是折扣因子在[0,1)，越小则关心当前奖励，越大关心未来奖励

**MDP**（Markov Decision Process,马尔可夫决策过程)：描述智能体如何在环境中连续做决策的框架；Markov Property指$$P(s_{t+1} \mid s_t) = P(s_{t+1} \mid s_0, \dots, s_t)$$未来只与当前状态有关，与更早的历史无关。



## Bellman equation 

State value，$$v_\pi(s) \doteq \mathbb{E}\left[G_t \mid S_t = s\right]=\mathbb{E}[R_{t+1}+\gamma G_{t+1} |S_t=s ]$$ 在策略π下，从状态s出发能够获得的期望长期回报，当$\pi$固定时为**policy evaluation**



贝尔曼公式描述了当前状态的state value=当前奖励 + 下一状态的state value
$$
\begin{aligned} G_t &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \\ &= R_{t+1} + \gamma\left(R_{t+2} + \gamma R_{t+3} + \dots\right) \\ &= R_{t+1} + \gamma G_{t+1}, 
\\
v_\pi(s) &\equiv \mathbb{E}\left[G_t \mid S_t = s\right] \\
&= \mathbb{E}\left[R_{t+1} + \gamma G_{t+1} \mid S_t = s\right] \\
&= \mathbb{E}\left[R_{t+1} \mid S_t = s\right] + \gamma \mathbb{E}\left[G_{t+1} \mid S_t = s\right].

\end{aligned}
$$

$$

$$

$$
\begin{aligned}
\mathbb{E}\left[R_{t+1} \mid S_t = s\right] &= \sum_{a \in \mathcal{A}} \pi(a \mid s)\,\mathbb{E}\left[R_{t+1} \mid S_t = s,\,A_t = a\right] \\
&= \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{r \in \mathcal{R}} p(r \mid s,a)\,r.
\\
\mathbb{E}\left[G_{t+1} \mid S_t = s\right] &= \sum_{s' \in \mathcal{S}} \mathbb{E}\left[G_{t+1} \mid S_t = s,\,S_{t+1} = s'\right]p(s' \mid s) \\ &= \sum_{s' \in \mathcal{S}} \mathbb{E}\left[G_{t+1} \mid S_{t+1} = s'\right]p(s' \mid s) \quad (\text{due to the Markov property}) \\ &= \sum_{s' \in \mathcal{S}} v_\pi(s')\,p(s' \mid s) \\ &= \sum_{s' \in \mathcal{S}} v_\pi(s') \sum_{a \in \mathcal{A}} p(s' \mid s,a)\,\pi(a \mid s).
\\

\end{aligned}
$$


$$
 \begin{aligned} v_\pi(s) &= \mathbb{E}\left[R_{t+1} \mid S_t = s\right] + \gamma \mathbb{E}\left[G_{t+1} \mid S_t = s\right], \\ &= \underbrace{\sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{r \in \mathcal{R}} p(r \mid s,a)\,r}_{\text{mean of immediate rewards}} + \gamma \underbrace{\sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} p(s' \mid s,a)\,v_\pi(s')}_{\text{mean of future rewards}} \\ &= \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ \sum_{r \in \mathcal{R}} p(r \mid s,a)\,r + \gamma \sum_{s' \in \mathcal{S}} p(s' \mid s,a)\,v_\pi(s') \right], \quad \text{for all } s \in \mathcal{S}. \end{aligned} 
$$




从矩阵的角度求解，定义
$$
\begin{aligned}
\\r_\pi(s) &\doteq \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{r \in \mathcal{R}} p(r \mid s,a)\,r, 
\\p_\pi(s' \mid s) &\doteq \sum_{a \in \mathcal{A}} \pi(a \mid s)\,p(s' \mid s,a)
\\
\end{aligned}
$$
前者是从状态s出发得到的immediate reward ，后者是从s出发到达其他状态的概率
$$
v_\pi(s_i) = r_\pi(s_i) + \gamma \sum_{s_j \in \mathcal{S}} p_\pi(s_j \mid s_i)\,v_\pi(s_j)
$$
公式对所有的状态都成立，那么有

![image-20260616184443312](assets/image-20260616184443312.png)



该式子可以通过**求逆**解决，**$$v_\pi = \left(I - \gamma P_\pi\right)^{-1} r_\pi$$**，但是当状态数较大时计算开销会增长较快

另一种方式是**迭代**，**$$v_{k+1} = r_\pi + \gamma P_\pi v_k,\quad k=0,1,2,\dots$$，**当k趋于无穷时，结果也就趋于求逆的结果



action value，$$q_\pi(s,a) = \mathbb{E}\left[G_t \mid S_t = s,\ A_t = a\right]$$，从状态s出发采取行动a能得到的长期回报。**在每个状态下选择最大的action value 不断迭代，一定能得到最优策略。**

与state value相关联$$v_\pi(s) = \sum_{a} \pi(a|s) q_\pi(s,a)$$，同时可以推出$$q_\pi(s,a) = \sum_{r\in\mathcal{R}} p(r|s,a)r + \gamma \sum_{s'\in\mathcal{S}} p(s'|s,a)v_\pi(s')$$



## Bellman optimal equation 

**state value可以衡量一个策略的好坏，当$$v_{\pi_1}(s) \ge v_{\pi_2}(s),\quad \text{for all } s\in\mathcal{S}$$，状态$\pi_1$更好**



**贝尔曼最优公式**是贝尔曼公式的一个特例，把求解最优策略的问题转换为求解最优state value的问题
$$
\begin{aligned} v(s) &= \max_{\pi(s)\in\Pi(s)} \sum_{a\in\mathcal{A}} \pi(a|s) \left( \sum_{r\in\mathcal{R}} p(r|s,a)r + \gamma \sum_{s'\in\mathcal{S}} p(s'|s,a)v(s') \right) \\ &= \max_{\pi(s)\in\Pi(s)} \sum_{a\in\mathcal{A}} \pi(a|s)q(s,a) \\&= f(v)= \max_{\pi\in\Pi}\big(r_\pi + \gamma P_\pi v\big)
\end{aligned}
$$
**Contraction mapping theorem，对于满足$$\left\| f(x_1) - f(x_2) \right\| \le \gamma \left\| x_1 - x_2 \right\|$$一定存在不动点$f(x^*)=x^*$,且$x^*$唯一，通过迭代$$x_{k+1} = f(x_k)$$一定能收敛到$x^*$**



## value iteration&policy iteration



### value iteration

**值迭代**初始随机一个$v_k$，基于**Contraction mapping theorem**的性质求解贝尔曼最优公式



值迭代可以拆解为

**Policy Update**，对于当前的$v_k$寻找最好的策略
$$
\pi_{k+1} = \underset{\pi}{\arg\max}\left(r^{\pi} + \gamma P^{\pi} v_k\right) \\
a^* = \underset{a}{\arg\max} Q_k(s,a)
\\
\pi_{k+1}(a|s)=
\begin{cases}
1, & a=a_k^*(s)\\
0, & a\neq a_k^*(s) & greedy 
\end{cases}
$$


**value update**	
$$
v_{k+1}(s) = \sum_{a}\pi_{k+1}(a|s)\underbrace{\left(\sum_{r}p(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v_{k}(s')\right)}_{q_{k}(s,a)}
\\
v_{k+1}(s) = \max_{a} q_{k}(s,a)
$$
当|$v_k-v_{k+1}$|差距很小时停止迭代



### policy iteration

初始随机一个策略$\pi_k$	



策略迭代可以拆解为

**policy evaluation**，求解state value 

$$
v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}
$$




**policy improvement**
$$
\pi_{k+1} = \underset{\pi}{\arg\max}\left(r_{\pi} + \gamma P_{\pi} v_{\pi_k}\right)
\\
\pi_{k+1}(s) = \underset{\pi}{\arg\max} \sum_{a} \pi(a|s) \underbrace{\left( \sum_{r} p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v_{\pi_k}(s') \right)}_{q_{\pi_k}(s,a)}
\\
a_k^{*}(s) = \underset{a}{\arg\max}\, q_{\pi_k}(a,s)
\\
\pi_{k+1}(a|s)=
\begin{cases}
1, & a=a_k^*(s),\\
0, & a\neq a_k^*(s).
\end{cases}
$$


### Truncated policy iteration

policy iteration和value iteration是Truncated policy iteration的两个特例，前者进行无穷次迭代精准求解$v_{\pi_k}$，后者只做一次bellman最优更新

Truncated policy iteration只做**m次 Bellman 期望更新**，不求精准的$v_{\pi_k}$，当m=1时为value iteration，m趋于无穷时policy iteration



## Monte Carlo Methods

蒙特卡洛方法是model-free，即不需要知道环境的状态转移概率和奖励函数（但是可以观察得到每一步的实际奖励）

Monte Carlo Estimation（蒙特卡洛估计），**在不知道环境模型和奖励函数的情况下，通过采样得到真实轨迹的回报来估计价值函数的方法**



### MC Basic

MC Basic是Policy iteration的变种

+ Policy Evaluation：对当前策略 $\pi_k$，**生成大量 episode**，计算每次访问 $(s,a)$ 后的回报： $$G_t = R_{t+1} + \gamma R_{t+2} + \dots$$ 并估计： $$Q(s,a) \approx q^{\pi_k}(s,a) = \mathbb{E}_{\pi_k}\left[G_t \mid S_t = s,\, A_t = a\right]$$，也就是进行Monte Carlo Estimation
+ Policy Improvement：对每个状态$$\pi_{k+1}(s) = \underset{a}{\arg\max}\ Q(s,a)$$



进行Monte Carlo Estimation时，不同的episode长度对策略有不同的影响，episode越长，估计的optimal state value越接近真值，策略越好。

<img src="assets/image-20260620094449501.png" alt="image-20260620094449501" style="zoom:67%;" />



### MC Exploring Starts

MC basic的效率较低，一个 episode 只更新起始的(s,a)，只有收集了大量的episode后才会更新策略

MC Exploring Starts是前者的改进，**二者的区别在于对样本的利用不同**，**一个 episode 更新沿途访问到的所有(s,a)，并且每个 episode 后就可以改进策略，因此样本利用率和学习效率更高。**在episode中，如果(s,a)出现了多次，first visit会只保留第一次出现得到的reward更新，every visit会对每一次取平均

![image-20260620100520197](assets/image-20260620100520197.png)



Exploring Starts要求每个 episode 从随机随机的 (s,a) **开始**，注意这里不是更新，那么N(s,a)→∞，MC basic也要求Exploring Starts



### MC ε-greedy

现实中做不到Exploring Starts，很多场景无法任意指定起始状态

ε-greedy 的思想是：**大部分时间选择当前最有动作，少部分时间随机探索**
$$
\pi(a|s)= \begin{cases} \displaystyle 1-\frac{\epsilon}{|\mathcal{A}(s)|}\big(|\mathcal{A}(s)|-1\big), & \text{for the greedy action},\\[6pt] \displaystyle \frac{\epsilon}{|\mathcal{A}(s)|}, & \text{for the other } |\mathcal{A}(s)|-1 \text{ actions}, \end{cases}
$$
$|\mathcal{A}(s)|$ 是状态数，ε ∈ [0, 1]，ε为0时，ε-greedy就是greedy，ε为1时，变为随即探索



MC ε-greedy与MC Exploring Starts相似，在policy improvement时有区别
$$
\pi_{k+1}(s)=\arg\max_{\pi\in\Pi_\epsilon}\sum_{a}\pi(a|s)q_{\pi_k}(s,a)

\\

\pi_{k+1}(a|s)=
\begin{cases}
\displaystyle 1-\frac{|\mathcal{A}(s)|-1}{|\mathcal{A}(s)|}\epsilon, & a=a_k^*,\\[6pt]
\displaystyle \frac{1}{|\mathcal{A}(s)|}\epsilon, & a\neq a_k^*,
\end{cases}
$$

## Stochastic Approximation



### Robbins-Monro algorithm

Robbins-Monro algorithm，随机逼近算法，**RM算法用于求解一个位置函数的根**，例如$g(w) = 0$; $g(x) $ 通常包含某个无法直接计算的期望，因此只能构造一个随机变量 $\hat g(w,η)$，满足
$$
\mathbb{E}\left[\tilde{g}(w,\eta) \mid w\right] = g(w)
$$




再通过迭代求解w
$$
w_{k+1}=w_{k}-a_{k}\tilde{g}(w_{k},\eta_{k}),\ k=1,2,3,\dots
\\
\tilde{g}(w,\eta)=g(w)+\eta,
$$
$\hat g$是无偏估计，$a_k$是大于0的系数



目标： $$w^* = E[X]$$ 

- 构造求根问题： $$g(w) = w - E[X]$$ 则 $$g(w^*) = 0.$$ 

- 由于 $E[X]$ 未知，只能观测 IID 样本 $x_k$，构造无偏估计： $$\tilde{g}(w_k,x_k) = w_k - x_k=(w_k-E(x))+(E(x)-x_k)=g(w)+\eta$$ ,因为 $$E[\tilde{g}(w_k,x_k)] = E[w_k - x_k] = w_k - E[X] = g(w_k)$$ ，这一步直接带入观测值(E(x)=x)到$\hat g$即可

- 代入 Robbins-Monro： $$w_{k+1} = w_k - a_k\tilde{g}(w_k,x_k)$$ 得到 $$w_{k+1} = w_k - a_k(w_k - x_k)$$ 即 $$\boxed{w_{k+1} = w_k + a_k(x_k - w_k)}$$





## Temporal-Difference Methods

Behavior Policy：用于收集数据的策略

Target Policy：**被**学习被评估的策略

on policy：Behavior Policy和Target Policy一致

off policy：反之

### TD learning of state values


$$
\underbrace{v_{t+1}(s_t)}_{\text{new estimate}}=\underbrace{v_t(s_t)}_{\text{current estimate}}-\alpha_t(s_t)\underbrace{\left[v_t(s_t)-\left(r_{t+1}+\gamma v_t(s_{t+1})\right)\right]}_{\text{TD error }\delta_t}
$$

+ $s_t$，t时刻的状态
+ $v_t(s_t)$ $s_t$在t时刻的state value 
+ $\alpha_t$系数 [0,1]
+ TD Target，$r_{t+1}+\gamma v_t(s_{t+1})$，执行动作后预测的state value 



这个公式用于求解state value，从RM算法出发有

+ 首先定义贝尔曼公式的新形式，$$v_\pi(s) = \mathbb{E}\left[R + \gamma G \mid S = s\right]=\mathbb{E}\left[R + \gamma v_\pi(S') \mid S = s\right]$$
+ 定义$$g(v(s)) = v(s) - \mathbb{E}\left[R + \gamma v_\pi(S') \mid s\right]$$，求解$$g(v(s)) = 0$$
+ $$ \begin{aligned} \tilde{g}(v(s)) &= v(s) - \left[r + \gamma v_\pi(s')\right] \\ &= \underbrace{\left(v(s) - \mathbb{E}\left[R + \gamma v_\pi(S') \mid s\right]\right)}_{g(v(s))} + \underbrace{\left(\mathbb{E}\left[R + \gamma v_\pi(S') \mid s\right] - \left[r + \gamma v_\pi(s')\right]\right)}_{\eta} \end{aligned} $$
+ $$ \begin{aligned} v_{k+1}(s) &= v_k(s) - \alpha_k \tilde{g}(v_k(s)) \\ &= v_k(s) - \alpha_k \left( v_k(s) - \left[ r_k + \gamma v_\pi(s_k') \right] \right), \quad k = 1,2,3,\dots \end{aligned} $$



| TD learning                                                  | MC learning                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **增量式更新**：TD 学习为增量算法，获取单条经验样本后即可立刻更新 | **非增量式更新**：MC 学习非增量，必须等待完整episode收集完毕才能更新 |
| **适用持续型任务**：依托增量更新特性，可同时处理分幕任务与无终止状态的持续型任务 | **仅适用分幕任务**：受非增量特性限制，仅能处理有限步数后终止的分幕任务 |
| **自举 (Bootstrapping)**：更新状态 / 动作价值时依赖价值的历史估计值，属于自举算法；因此需要预先给定价值初始猜测 | **无自举**：蒙特卡洛不使用自举，无需初始估值，可直接利用完整回报估计状态 / 动作价值 |
| **估计方差更低**：TD 估计方差小于 MC，涉及随机变量更少。以 Sarsa 估计动作价值qπ(st,at)为例，仅需Rt+1,St+1,At+1三类随机样本 | **估计方差更高**：涉及大量随机变量，方差更大。估计qπ(st,at)需要Rt+1+γRt+2+γ2Rt+3+…；假设一幕长度为L、每个状态有 $ |



###  TD learning of action values

SARSA，求解action value，用到了五元组$$(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$$
$$
q_{t+1}(s_t,a_t) = q_t(s_t,a_t) - \alpha_t(s_t,a_t)\left[ q_t(s_t,a_t) - \big(r_{t+1} + \gamma q_t(s_{t+1},a_{t+1})\big) \right]
$$
SARSA是on policy，$\pi$求解得到experience，再用experience更新$\pi$



### Q-learning 

Q-learning求解一个**贝尔曼最优公式**：$$q(s,a) = \mathbb{E}\left[ R_{t+1} + \gamma \max_{a} q(S_{t+1},a) \,\bigg|\, S_t=s,A_t=a \right]$$

期望无法直接求解，用一次采样作为样本估计，再通过RM算法随机逼近
$$
q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t) \left[ q_t(s_t, a_t) - \left( r_{t+1} + \gamma \max_{a \in A(s_{t+1})} q_t(s_{t+1}, a) \right) \right]
$$
Q-learning是off-policy，但也可以是on-policy的，区分在于是否是ε-greedy：如果是，那么生成的策略会同时用来收集数据，就符合on policy





## Value Function Methods

核心思想是用函数拟合$v_{\pi}(s)$，节省存储，增强泛化能力


$$
J(w) = \mathbb{E}\left[\left(v_\pi(S) - \hat{v}(S,w)\right)^2\right]
$$
对于期望的求解有两种方法



**Uniform Distribution**认为所有状态同等重要$$J(w) = \frac{1}{|\mathcal{S}|}\sum_s \left(v_\pi(s) - \hat{v}(s,w)\right)^2$$

**Stationary Distribution**定义$$J(w) = \sum_{s} d^\pi(s) \left(v_\pi(s) - \hat{v}(s,w)\right)^2$$，$$d^\pi = d^\pi P_\pi$$ 是智能体长期按照策略$\pi$运行时，处于状态s的概率



随后求解
$$
w_{k+1} = w_k - \alpha_k \nabla_w J(w_k)
\\
\nabla_w J(w) = -2\mathbb{E}\left[\left(v_\pi(S) - \hat{v}(S,w)\right)\nabla_w \hat{v}(S,w)\right]
\\
w_{k+1} = w_k + 2\alpha_k \mathbb{E}\left[\left(v_\pi(S) - \hat{v}(S,w_k)\right)\nabla_w \hat{v}(S,w_k)\right]
$$
这里的期望通常无法求出，通过SGD的思想取一个样本近似有
$$
w_{t+1} = w_t + \alpha_t \left(v_\pi(s_t) - \hat{v}(s_t,w_t)\right)\nabla_w \hat{v}(s_t,w_t)
$$
这里的$v_{\pi}(s)$无法得到，有两种方法解决

+ Monte Carlo：$v_{\pi}(s)$=$$g_t = R_{t+1} + \gamma R_{t+2} + \dots$$，得到$$w_{t+1}=w_t+\alpha_t\big(g_t-\hat{v}(s_t,w_t)\big)\nabla_w\hat{v}(s_t,w_t)$$
+ Temporal Difference：$$v_{\pi}(s)=r_{t+1} + \gamma \hat{v}(s_{t+1}, w_t)$$，得到$$w_{t+1}=w_t+\alpha_t\big(r_{t+1}+\gamma\hat{v}(s_{t+1},w_t)-\hat{v}(s_t,w_t)\big)\nabla_w\hat{v}(s_t,w_t)$$



$\hat v(s,t)$的近似函数可以是线性的也可以是神经网络	



###Sarsa & Q-learning with function approximation

Sarsa，对 action value 近似
$$
w_{t+1}=w_t+\alpha_t\left[r_{t+1}+\gamma\hat{q}(s_{t+1},a_{t+1},w_t)-\hat{q}(s_t,a_t,w_t)\right]\nabla_w\hat{q}(s_t,a_t,w_t)
$$
Q-learning，对optimal action value 近似
$$
w_{t+1}=w_t+\alpha_t\left[r_{t+1}+\gamma\max_{a\in\mathcal{A}(s_{t+1})}\hat{q}(s_{t+1},a,w_t)-\hat{q}(s_t,a_t,w_t)\right]\nabla_w\hat{q}(s_t,a_t,w_t)
$$


### DQN

DQN用神经网络近似Q函数，目的是优化下面的式子
$$
J \equiv \mathbb{E}\left[\left(R+\gamma \max_{a\in\mathcal{A}(S')}\hat{q}(S',a,w)-\hat{q}(S,A,w)\right)^2\right]
\\
J = \mathbb{E}\left[\left(y - \hat{q}(S,A,w)\right)^2\right]
$$


DQN公式中的w出现在两个位置，不好直接求梯度；DQN引入**两个神经网络**：主网络 $$\hat{q}(s,a,w)$$，目标网络 $$\hat{q}(s,a,w_T)$$，目标网络中的参数$w_T$在一段时间内固定，因此问题变为标准的监督学习；在一段时间后，用w更新目标网络，随后重复流程。



DQN中存在期望因此目标函数必须服从某个概率分布，这里默认是服从均匀分布的，因为没有先验知识。但是实际采样不是均匀的，真实的分布与策略有关，并且样本之间强相关，不满足独立同分布。为此，DQN提出 **Replay Buffer**，经验回放，将所有的经验存储起来，训练时随机抽取。



## Policy Gradient Methods

用函数$\pi(a \mid s; \theta)$代替表格来表示策略，一般是神经网络，这种方法不能直接修改策略，只能修改参数$\theta$



Policy Gradient中$\theta$会同时影响很多状态，可能无法找到一个参数使所有的状态价值同时达到最大，这里将所有状态压缩成一个标量目标$J(\theta)$，定义最优策略为$$\max_{\theta} J(\theta)$$



目标函数的定义有两种

### Average state value

首先是Average state value
$$
\bar{v}^{\pi} \equiv \sum_{s \in S} d(s) v^{\pi}(s)
$$
其中$$d(s) \ge 0,\quad \sum d(s) = 1$$，为概率分布，可以是固定分布或平稳分布(Stationary Distribution)

另一种形式为
$$
 \begin{aligned} \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^{t} R_{t+1}\right] &= \sum_{s \in \mathcal{S}} d(s) \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^{t} R_{t+1} \mid S_{0}=s\right] \\ &= \sum_{s \in \mathcal{S}} d(s) v_{\pi}(s) \\ &= \bar{v}_{\pi}. \end{aligned} 
$$

### Average reward

第二种是Average reward
$$
\bar{r}_{\pi} \triangleq \sum_{s \in \mathcal{S}} d_{\pi}(s) r_{\pi}(s)

\\
r_{\pi}(s) \triangleq \sum_{a \in \mathcal{A}} \pi(a \mid s, \theta) r(s, a)
$$


另一种形式为
$$
\lim_{n \to \infty} \frac{1}{n}\mathbb{E}\left[\sum_{t=0}^{n-1} R_{t+1}\right] = \sum_{s \in \mathcal{S}} d_{\pi}(s) r_{\pi}(s) \equiv \bar{r}_{\pi}
$$


可以证明$$\bar{r}_{\pi} = (1 - \gamma)\bar{v}_{\pi}.$$

### gradient

目标函数的梯度可以统一为
$$
\begin{aligned}
\nabla_{\theta} J(\theta) &= \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \nabla_{\theta} \pi(a \mid s, \theta) q_{\pi}(s, a) \\
&= \mathbb{E}_{S \sim \eta}\left[\sum_{a \in \mathcal{A}} \nabla_{\theta} \pi(a \mid S, \theta) q_{\pi}(S, a)\right]
\\
&= \mathbb{E}\left[\sum_{a \in \mathcal{A}} \pi(a \mid S, \theta) \nabla_{\theta} \ln \pi(a \mid S, \theta) q_{\pi}(S, a)\right] \\
&= \mathbb{E}_{S \sim \eta,\, A \sim \pi(S,\theta)}\left[\nabla_{\theta} \ln \pi(A \mid S, \theta) q_{\pi}(S, A)\right]
\end{aligned}
$$

$$
\nabla_{\theta} \ln \pi(a \mid s, \theta) = \frac{\nabla_{\theta} \pi(a \mid s, \theta)}{\pi(a \mid s, \theta)}
$$

为了满足对数函数的性质（x>0），定义
$$
\pi(a \mid s, \theta) = \frac{e^{h(s,a,\theta)}}{\sum_{a' \in \mathcal{A}} e^{h(s,a',\theta)}} \ h函数是在s选择a的概率
$$


再次变化可得$$ \theta_{t+1} = \theta_t + \alpha \underbrace{\left( \frac{q_t(s_t,a_t)}{\pi(a_t \mid s_t,\theta_t)} \right)}_{\beta_t} \nabla_{\theta}\pi(a_t \mid s_t,\theta_t) $$

经过证明有
$$
\begin{aligned} \pi(a_t \mid s_t, \theta_{t+1}) &\approx \pi(a_t \mid s_t, \theta_t) + \big(\nabla_{\theta}\pi(a_t \mid s_t, \theta_t)\big)^T (\theta_{t+1} - \theta_t) \\ &= \pi(a_t \mid s_t, \theta_t) + \alpha \beta_t \big(\nabla_{\theta}\pi(a_t \mid s_t, \theta_t)\big)^T \big(\nabla_{\theta}\pi(a_t \mid s_t, \theta_t)\big) \\ &= \pi(a_t \mid s_t, \theta_t) + \alpha \beta_t \left\lVert \nabla_{\theta}\pi(a_t \mid s_t, \theta_t) \right\rVert_2^2. \end{aligned}
$$
可以发现$\beta_t$>=0，$\pi(a_t \mid s_t, \theta_{t+1})>\pi(a_t \mid s_t, \theta_{t})$，即action value越大，执行相应策略的可能就会越大，体现了exploitation;对应状态的概率越小，执行的相应策略的可能就会越大，体现了exploration.



梯度的式子中期望无法直接计算，会使用采样近似，式子中的$q_{\pi}$使用**MC**的方法估计，即采样一个episode，对每一个时间步更新，也就是reinforce算法，因为使用MC的方法，所以必须等一个episode采样完才能更新参数



