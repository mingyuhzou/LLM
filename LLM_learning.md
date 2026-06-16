# LLM



# RL

强化学习的目标是**求解最优策略**



## 基本概念

**State**：环境给agent的信息

**Action**：agent在某个状态下可以执行的决策

**Policy**: agent在某个状态下选择动作的原则，$$\pi\left(a_{2}\mid s_{1}\right)=1$$表示在s1的状态下采取动作a2的概率为1	

**reward**：$$p\left(r=-1 \mid s_{1}, a_{1}\right)=1$$在状态s1和a1下得到分数-1的概率为1

**Trajectory**：state-action-reward链 $$ S_1 \xrightarrow[\,r=0\,]{a_2} S_2 \xrightarrow[\,r=0\,]{a_3} S_5 \xrightarrow[\,r=0\,]{a_3} S_8 \xrightarrow[\,r=1\,]{a_2} S_9. $$ 该链上的reward之和为return，return可用于评估policy；存在Terminal state的Trajectory称为eposide

**Discounted** reward：$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$$，用于解决无限发散的序列，γ是折扣因子在[0,1)，越小则关心当前奖励，越大关心未来奖励

**MDP**（Markov Decision Process,马尔可夫决策过程)：描述智能体如何在环境中连续做决策的框架；Markov Property指$$P(s_{t+1} \mid s_t) = P(s_{t+1} \mid s_0, \dots, s_t)$$未来至于当前状态有关，与更早的历史无关。



## 贝尔曼公式

State value，$$v_\pi(s) \doteq \mathbb{E}\left[G_t \mid S_t = s\right]$$ 在策略π下，从状态s出发能够获得的期望长期回报，**衡量一个状态的好坏**，当$\pi$固定时为policy evaluation



贝尔曼公式描述了当前状态的state value=当前奖励+下一状态的state value
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



该式子可以通过**求逆**解决，$$v_\pi = \left(I - \gamma P_\pi\right)^{-1} r_\pi$$，但是当状态数较大时计算开销会增长较快

另一种方式是**迭代**，$$v_{k+1} = r_\pi + \gamma P_\pi v_k,\quad k=0,1,2,\dots$$，当k趋于无穷时，结果也就趋于求逆的结果

