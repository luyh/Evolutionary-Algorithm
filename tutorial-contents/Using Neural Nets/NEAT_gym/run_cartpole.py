"""
Using NEAT for reinforcement learning.

The detail for NEAT can be find in : http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""
import neat
import numpy as np
import gym
import visualize

GAME = 'CartPole-v0'
env = gym.make(GAME).unwrapped

print("action space: {0!r}".format(env.action_space))
print("observation space: {0!r}".format(env.observation_space))

CONFIG = "./config"
EP_STEP = 500           # maximum episode steps
GENERATION_EP = 10      # evaluate by the minimum of 10-episode rewards
TRAINING = True         # training or testing
winner = "winner-169.bin"
CHECKPOINT = 9          # test on this checkpoint


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        ep_r = []
        for ep in range(GENERATION_EP): # run many episodes for the genome in case it's lucky
            accumulative_r = 0.         # stage longer to get a greater episode reward
            observation = env.reset()
            for t in range(EP_STEP):
                action_values = net.activate(observation)
                action = np.argmax(action_values)
                observation_, reward, done, _ = env.step(action)
                accumulative_r +=  (4.8 - abs(observation_[0]) * reward)
                if done:
                    break
                observation = observation_
            ep_r.append(accumulative_r)
        genome.fitness = np.min(ep_r)/float(EP_STEP)    # depends on the minimum episode reward

import pickle
def train():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG)
    pop = neat.Population(config)

    # recode history
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(5))

    winner = pop.run(eval_genomes, 10)       # train 10 generations

    with open("winners/winner-{}.bin".format(winner.key), "wb") as f:
        pickle.dump(winner, f, 2)

    # visualize training
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    return winner


def evaluation(winner):

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%i' % CHECKPOINT)
    # winner = p.run(eval_genomes, 1)     # find the winner in restored population

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG)

    if not TRAINING:
        with open("winners/{}".format(winner), "rb") as f:
            winner = pickle.load(f)

    # show winner net
    node_names = {-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    while True:
        data = []
        s = env.reset()
        for t in range(EP_STEP):
            env.render()
            av = net.activate(s)
            a = np.argmax(av)
            s, r, done, _ = env.step(a)
            data.append(np.hstack((s, a, (4.8 - abs(s[0]) * r))))
            if done: break

        data = np.array(data)
        score = np.sum(data[:, -1]) / EP_STEP

        print('score',score)
        print('data', data)


if __name__ == '__main__':
    if TRAINING:
        print('TRAINING:')
        winner = train()
    print('evaluation:')
    evaluation(winner)
