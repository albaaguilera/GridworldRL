# GridworldRL
Single-agent Reinforcement Learning in a simple Grid-world environment. 

Our agent, represented as a blue dot, needs to explore the environment to find a target located randomly each time. By trial and error, the agent finds the most optimal way to find the target.

We compare **Value Iteration** (known transitions and rewards: model-based) with **Q-learning** (unknown transitions and rewards: model-free).

| **Feature**                         | **Value Iteration**                                                                                                                                     | **Q-learning**                                                                                          |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Needs full knowledge of the environment? | ✅ Yes                                                                                                                                              | ❌ No                                                                                                     |
| Exploration required?               | ❌ No                                                                                                                                               | ✅ Yes (e.g., ε-greedy)                                                                                  |
| Algorithm                           | Bellman equation updates over all states                                         | Trial-and-error with q-table updates over all states|
| Convergence                         | Fast & deterministic                                                                                                                   | Slower, stochastic                                                                                       |
| When to use                         | Known small MDPs                                                                                                                                       | Large or unknown environments                                                                            |

---

## 🎥 Single-Agent Behavior

<table>
<tr>
<td align="center"><strong>Value Iteration</strong><br><img src="output/value_iter_run.mp4" width="300"/></td>
<td align="center"><strong>Q-learning</strong><br><img src="output/qlearning_run.mp4" width="300"/></td>
</tr>
</table>

## 🧪 Training Results

![Training Results](data/training_results.png)