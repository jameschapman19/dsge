def evaluate_classical(env):
    env.solve()
    print(f"total utility: {env.total_utility(env.c)}")
    env.render()
