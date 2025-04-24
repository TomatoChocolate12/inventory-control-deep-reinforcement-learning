# inventory-control-deep-reinforcement-learning
applying deep reinforcement learning on the problem of inventory management

enviroment/problem setting:

# -----------------------
# 3. Environment & Cost
# -----------------------
p, a_cost, b_cost = 1, 1, 2
def inventory_cost(s_next, a):
    if s_next >= 0:
        return p * a + a_cost * s_next
    else:
        return p * a + b_cost * -s_next

# -----------------------
# 4. Hyperparameters
# -----------------------
num_actions        = 3
T                  = 100
min_inventory      = -10
max_inventory      = 10
state_dim          = 2       # [inventory_level, time_step]
buffer_capacity    = 5000
batch_size         = 32
gamma              = 1
epsilon            = 0.1
target_update_freq = 100
learning_rate      = 1e-3
num_episodes       = 500

   # step environment (for now we are considering unifom demand)
        w       = random.randint(0, 10)
        s_next  = s + a - w (planck eqn of MDP)
        cost    = inventory_cost(s_next, a) 
    