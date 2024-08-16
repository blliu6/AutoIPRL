def reappear(agent, env):
    env.vis = False
    agent.epsilon = 1
    episode_return = 0
    state, info = env.reset()
    done, truncated = False, False
    while not done and not truncated:
        action = agent.take_action(state, len(env.action))
        next_state, reward, done, truncated, info = env.step(action)
        episode_return += reward
        state = next_state
    print(f'episode_return:{episode_return}')
