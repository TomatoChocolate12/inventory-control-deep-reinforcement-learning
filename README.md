# Inventory Control using Deep Reinforcement Learning

This project applies deep reinforcement learning techniques to solve inventory management problems.

## Environment Setup

The environment simulates an inventory management system with the following characteristics:

### Cost Structure
- Purchasing cost (p) = 1
- Holding cost (a_cost) = 1
- Shortage cost (b_cost) = 2

```python
def inventory_cost(s_next, a):
    if s_next >= 0:
        return p * a + a_cost * s_next    # Holding cost
    else:
        return p * a + b_cost * -s_next   # Shortage cost
```

### System Dynamics
- State transitions follow: s_next = s + a - w
- Where:
  - s: current inventory level
  - a: ordering action
  - w: random demand (uniform distribution between 0 and 10)

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| num_actions | 3 | Available ordering actions |
| T | 100 | Time horizon |
| min_inventory | -10 | Minimum inventory level |
| max_inventory | 10 | Maximum inventory level |
| state_dim | 2 | Dimension of state [inventory_level, time_step] |
| buffer_capacity | 5000 | Experience replay buffer size |
| batch_size | 32 | Training batch size |
| gamma | 1 | Discount factor |
| epsilon | 0.1 | Exploration rate |
| target_update_freq | 100 | Target network update frequency |
| learning_rate | 1e-3 | Learning rate |
| num_episodes | 500 | Number of training episodes |