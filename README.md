# GridworldRL
Single-agent and Multi-agent Reinforcement Learning in a simple Grid-world environment. 

Compared **Value Iteration** (known transitions and rewards: model-based) with **Q-learning** (unknown transitions and rewards: model-free).

| Metric                        | Value Iteration | Q-learning |
|-------------------------------|-----------------|------------|
| Episodes to convergence       | *1*             | *~120*     |
| Average reward                | *1.0*           | *0.9*      |
| Average steps to goal         | *~6*            | *~6.8*     |

| **Feature**                         | **Value Iteration**                                                                                                                                     | **Q-learning**                                                                                          |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Needs full knowledge of the environment? | ✅ Yes                                                                                                                                              | ❌ No                                                                                                     |
| Algorithm                           | Bellman updates over all states:<br>**V(s) = max<sub>a</sub> [ R(s,a) + γ Σ P(s'|s,a) * V(s') ]**                                                    | Trial-and-error with updates:<br>**Q(s,a) ← Q(s,a) + α [ r + γ * max<sub>a'</sub> Q(s',a') - Q(s,a) ]** |
| Exploration required?               | ❌ No                                                                                                                                               | ✅ Yes (e.g., ε-greedy)                                                                                  |
| Convergence                         | Fast & deterministic                                                                                                                   | Slower, stochastic                                                                                       |
| When to use                         | Known small MDPs                                                                                                                                       | Large or unknown environments                                                                            |

---

## 🎥 Single-Agent Behavior

<table>
<tr>
<td align="center"><strong>Value Iteration</strong><br><img src="gifs/value_iteration.gif" width="300"/></td>
<td align="center"><strong>Q-learning</strong><br><img src="gifs/qlearning.gif" width="300"/></td>
</tr>
</table>


## 🎥 Multi-Agent Behavior

<table>
<tr>
<td align="center"><strong>Value Iteration</strong><br><img src="gifs/value_iteration.gif" width="300"/></td>
<td align="center"><strong>Q-learning</strong><br><img src="gifs/qlearning.gif" width="300"/></td>
</tr>
</table>


## 🧪 Training Results

| Metric                        | Value Iteration | Q-learning |
|-------------------------------|-----------------|------------|
| Episodes to convergence       | *1*             | *~120*     |
| Average reward                | *1.0*           | *0.9*      |
| Average steps to goal         | *~6*            | *~6.8*     |

Cummulative reward : average cumulative reward per episode over time
Convergence time or episodes to convergence
steps per episode