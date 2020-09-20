import gym

def main():
    env = gym.make('Blackjack-v0')
    env.reset()
    done = False
    print(render_obs(env._get_obs()))
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        action_str = "hit" if action else "stay"
        print(f"{action_str}, {render_obs(obs)}")
    print("win" if reward > 0 else "lose")
    env.close()

def render_obs(obs):
    return f"sum: {obs[0]}, dealer: {obs[1]}, usable ace?: {obs[2]}"

if __name__ == "__main__":
    main()
