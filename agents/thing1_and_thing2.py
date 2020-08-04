import collections

import numpy as np
import pommerman
import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from pommerman import agents
from pommerman import characters
from pommerman.agents import BaseAgent

nest = tf.contrib.framework.nest
LSTM_SIZE = 128

AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')
StepOutputInfo = collections.namedtuple('StepOutputInfo',
                                        'episode_return episode_step')
StepOutput = collections.namedtuple('StepOutput',
                                    'reward info done observation')


class Agent(snt.RNNCore):
    """Agent with ResNet."""

    def __init__(self, num_actions):
        super(Agent, self).__init__(name='agent')

        self._num_actions = num_actions

        with self._enter_variable_scope():
            self._core = tf.contrib.rnn.LSTMBlockCell(LSTM_SIZE)

    def initial_state(self, batch_size):
        return self._core.zero_state(batch_size, tf.float32)

    def _torso(self, input_):
        last_action, env_output = input_
        reward, _, _, observations = env_output

        # board, feature = observations

        board, bomb_blast_strength, bomb_life, feature = observations

        board = tf.one_hot(
            board,
            14,
            name='one_hot_board'
        )

        board = tf.concat([board, tf.expand_dims(bomb_blast_strength, axis=-1), tf.expand_dims(bomb_life, axis=-1)],
                          axis=-1)

        board_shape = tf.shape(board)
        batch_size = board_shape[0]
        x_dim = board_shape[1]
        y_dim = board_shape[2]
        f_dim = board_shape[3]

        feature_size = tf.shape(feature)[1]

        conv_out = snt.Conv2D(16, 3, stride=1, padding='VALID')(board)
        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.Conv2D(32, 3, stride=1, padding='VALID')(conv_out)
        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.Conv2D(64, 3, stride=1, padding='VALID')(conv_out)
        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.Conv2D(128, 3, stride=1, padding='VALID')(conv_out)
        conv_out = tf.nn.relu(conv_out)
        f = tf.reshape(conv_out, [batch_size, 128])

        # conv_out = snt.Conv2D(256, 3, stride=1, padding='VALID')(conv_out)
        # conv_out = tf.nn.relu(conv_out)

        f = tf.concat([f, feature], axis=-1)

        # Append clipped last reward and one hot last action.
        clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
        last_action_list = [last_action[:, i] for i in range(len(self._num_actions))]
        one_hot_last_action = tf.concat([tf.one_hot(a, n) for a, n in zip(last_action_list, self._num_actions)],
                                        axis=-1)
        return tf.concat([f, clipped_reward, one_hot_last_action], axis=1)

    def _head(self, core_output):
        logits = snt.Linear(sum(self._num_actions))(
            core_output)

        policy_logits = tf.split(logits, self._num_actions, axis=-1, name='policy_logits')
        baseline = tf.squeeze(snt.Linear(1, name='baseline')(core_output), axis=-1)

        # Sample an action from the policy.
        new_action = tf.concat(list(map(lambda logits: tf.multinomial(logits, num_samples=1,
                                                                      output_dtype=tf.int32), policy_logits)), -1,
                               name='new_action')
        # new_action = tf.squeeze(new_action, 1, name='new_action')

        return AgentOutput(new_action, policy_logits, baseline)

    def _build(self, input_, core_state):
        action, env_output = input_
        actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                  (action, env_output))
        outputs, core_state = self.unroll(actions, env_outputs, core_state)
        return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs), core_state

    @snt.reuse_variables
    def unroll(self, actions, env_outputs, core_state):
        _, _, done, _ = env_outputs

        torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))

        # Note, in this implementation we can't use CuDNN RNN to speed things up due
        # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
        # changed to implement snt.LSTMCell).
        initial_core_state = self._core.zero_state(tf.shape(actions)[1], tf.float32)
        core_output_list = []
        for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
            # If the episode ended, the core state should be reset before the next.
            core_state = nest.map_structure(functools.partial(tf.where, d),
                                            initial_core_state, core_state)
            if True:
                core_output, core_state = self._core(input_, core_state)
                core_output_list.append(core_output)
            else:
                core_output_list.append(input_)

        return snt.BatchApply(self._head)(tf.stack(core_output_list)), core_state


class ScalableAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def __init__(self, agent=None, character=characters.Bomber, checkpoint_dir='agents/pretrained', agent_num=0,
                 printing=False, disable_message=False, old=False):
        super(ScalableAgent, self).__init__(character)

        self.printing = printing
        self.disable_message = disable_message

        self.old = old

        with tf.Graph().as_default():

            if self.old:
                for i in range(agent_num + 1):
                    agent = Agent((6, 8, 8))
                self.observation = (tf.placeholder(tf.int32, [9, 9]), tf.placeholder(tf.float32, [9, 9]),
                                    tf.placeholder(tf.float32, [9, 9]), tf.placeholder(tf.float32, [22]))

            else:
                with tf.variable_scope('agent_' + str(agent_num)):
                    agent = Agent((6, 8, 8))
                self.observation = (tf.placeholder(tf.int32, [9, 9]), tf.placeholder(tf.float32, [9, 9]),
                                    tf.placeholder(tf.float32, [9, 9]), tf.placeholder(tf.float32, [23]))

            # self.session = session

            reward = tf.constant(0.)
            done = tf.constant(False)
            info = StepOutputInfo(tf.constant(0.), tf.constant(0))
            self.last_action = tf.placeholder(tf.int32, [1, 3])

            self.core_state = tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, [1, LSTM_SIZE]),
                                                            tf.placeholder(tf.float32, [1, LSTM_SIZE]))

            env_output = StepOutput(
                reward,
                info,
                done,
                self.observation)

            batched_env_output = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                                    env_output)
            self.agent_output, self.agent_state = agent((self.last_action, batched_env_output), self.core_state)

            self.state = (np.zeros((1, LSTM_SIZE), dtype=np.float32), np.zeros((1, LSTM_SIZE), dtype=np.float32))
            self.previous_action = np.zeros((1, 3), dtype=np.int32)

            self.session = tf.train.SingularMonitoredSession(checkpoint_dir=checkpoint_dir,
                                                             config=tf.ConfigProto(device_count={'GPU': 0}))

    def _parse_observation(self, obs):
        agent = obs["board"][obs["position"][0]][obs["position"][1]]
        teammate = obs["teammate"].value
        enemies = [e.value for e in obs["enemies"][:2]]

        centered_board = np.ones((9, 9), dtype=np.int32)
        centered_bomb_blast_strength = np.ones((9, 9), dtype=np.float32)
        centered_bomb_life = np.ones((9, 9), dtype=np.float32)

        x, y = obs["position"]
        centered_board[max(0, 4 - x):min(9, 15 - x), max(0, 4 - y):min(9, 15 - y)] = obs["board"][
                                                                                     max(0, x - 4):min(11, x + 5),
                                                                                     max(0, y - 4):min(11, y + 5)]

        centered_board_copy = np.copy(centered_board)
        centered_board[centered_board_copy == agent] = 10
        centered_board[centered_board_copy == teammate] = 11
        centered_board[centered_board_copy == enemies[0]] = 12
        centered_board[centered_board_copy == enemies[1]] = 13

        centered_bomb_blast_strength[max(0, 4 - x):min(9, 15 - x), max(0, 4 - y):min(9, 15 - y)] = obs[
                                                                                                       "bomb_blast_strength"].astype(
            np.float32)[
                                                                                                   max(0,
                                                                                                       x - 4):min(
                                                                                                       11, x + 5),
                                                                                                   max(0,
                                                                                                       y - 4):min(
                                                                                                       11, y + 5)]

        centered_bomb_life[max(0, 4 - x):min(9, 15 - x), max(0, 4 - y):min(9, 15 - y)] = obs["bomb_life"].astype(
            np.float32)[
                                                                                         max(0, x - 4):min(11,
                                                                                                           x + 5),
                                                                                         max(0, y - 4):min(11,
                                                                                                           y + 5)]

        position = np.array(obs["position"], dtype=np.float32) / 10 * 2 - 1
        ammo = np.array([obs["ammo"]], dtype=np.float32)
        blast_strength = np.array([obs["blast_strength"]], dtype=np.float32)
        can_kick = np.array([obs["can_kick"]], dtype=np.float32)
        if self.old:
            teammate_alive = np.array([teammate in ["alive"]], dtype=np.float32)
        else:
            teammate_alive = np.array([teammate in obs["alive"]], dtype=np.float32)
        two_enemies = np.array([enemies[0] in obs["alive"] and enemies[1] in obs["alive"]], dtype=np.float32)

        # print(teammate_alive, two_enemies)

        message = np.zeros((2, 8), dtype=np.float32)
        message[np.arange(2), obs["message"]] = 1
        message = message.reshape(-1)

        if self.old:
            feature = np.concatenate([position, ammo, blast_strength, can_kick, teammate_alive, message])
        else:
            feature = np.concatenate([position, ammo, blast_strength, can_kick, teammate_alive, two_enemies, message])

        return (centered_board, centered_bomb_blast_strength, centered_bomb_life, feature)

    def act(self, obs, action_space):
        if self.disable_message:
            obs["message"] = (0, 0)

        if self.printing:
            if obs["message"][0] != 0 or obs["message"][1] != 0:
                print(obs["message"])

        parsed_observation = self._parse_observation(obs)
        output, state = self.session.run([self.agent_output, self.agent_state],
                                         feed_dict={self.observation: parsed_observation,
                                                    self.last_action: self.previous_action,
                                                    self.core_state: self.state})

        # print(obs["message"])

        if self.printing:
            dist_index = 2
            probs = (np.exp(output[1][dist_index][0]) / sum(np.exp(output[1][dist_index][0])))
            # print('|' + '|'.join(str(i) * int(prob * 100) for i, prob in enumerate(probs)) + '|')
            # print(tuple(output[0][0])[dist_index])
            # if output[0][0][1] != 0 or output[0][0][2] != 0:
            #    print(output[0][0][1], output[0][0][2])

        action = tuple(output[0][0])
        self.previous_action = output[0]
        self.state = state

        return tuple(int(a) for a in action)

    def episode_end(self, reward):
        self.state = (np.zeros((1, LSTM_SIZE), dtype=np.float32), np.zeros((1, LSTM_SIZE), dtype=np.float32))
        self.previous_action = np.zeros((1, 3), dtype=np.int32)


import sys, os


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


def save_video(env, output='videos/output.avi'):
    import cv2, pyglet

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, fourcc, 10.0, (738, 620))

    MESSAGE_COLORS = [(255, 255, 255),
                      (255, 0, 0),
                      (255, 127, 0),
                      (255, 255, 0),
                      (0, 255, 0),
                      (0, 0, 255),
                      (75, 0, 130),
                      (148, 0, 211)]
    messages = [None, None, None, None]
    cooldowns = [0, 0, 0, 0]

    state = env.reset()
    done = False
    while True:
        env.render(do_sleep=False)

        pyglet.image.get_buffer_manager().get_color_buffer().save('videos/temp.png')
        im = cv2.imread('videos/temp.png')

        if done:
            out.write(im)
            break

        blockPrint()
        actions = env.act(state)
        enablePrint()

        for i in range(4):
            if type(actions[i]) != tuple:
                continue
            if actions[i][1:] != (0, 0):
                print(actions[i][1:])
                messages[i] = actions[i][1:]
                cooldowns[i] = 10

            cooldowns[i] -= 1
            if cooldowns[i] <= 0:
                messages[i] = None

            if messages[i] is not None:
                cx = state[i]['position'][1] * 50 + 20 + 32
                cy = state[i]['position'][0] * 50 + 50 + 12

                cv2.circle(im, (cx, cy), 8, (200, 200, 200), thickness=-1, lineType=cv2.LINE_AA, shift=0)
                cv2.circle(im, (cx + 16, cy), 8, (200, 200, 200), thickness=-1, lineType=cv2.LINE_AA, shift=0)
                cv2.rectangle(im, (cx, cy - 8), (cx + 14, cy + 8), (200, 200, 200), thickness=-1)
                cv2.fillConvexPoly(im, np.int_([(cx - 6, cy), (cx - 8, cy + 20), (cx, cy + 4)]), (200, 200, 200),
                                   lineType=cv2.LINE_AA, shift=0)
                cv2.circle(im, (cx, cy), 5, MESSAGE_COLORS[messages[i][0]], thickness=-1, lineType=cv2.LINE_AA, shift=0)
                cv2.circle(im, (cx + 14, cy), 5, MESSAGE_COLORS[messages[i][1]], thickness=-1, lineType=cv2.LINE_AA,
                           shift=0)

                cx = 600 + 68
                cy = int(i * 147.5) + 36

                cv2.circle(im, (cx, cy), 15, (200, 200, 200), thickness=-1, lineType=cv2.LINE_AA, shift=0)
                cv2.circle(im, (cx + 30, cy), 15, (200, 200, 200), thickness=-1, lineType=cv2.LINE_AA, shift=0)
                cv2.rectangle(im, (cx, cy - 15), (cx + 30, cy + 15), (200, 200, 200), thickness=-1)
                cv2.circle(im, (cx, cy), 10, MESSAGE_COLORS[messages[i][0]], thickness=-1, lineType=cv2.LINE_AA,
                           shift=0)
                cv2.circle(im, (cx + 30, cy), 10, MESSAGE_COLORS[messages[i][1]], thickness=-1, lineType=cv2.LINE_AA,
                           shift=0)

        out.write(im)

        state, reward, done, info = env.step(actions)

    out.release()


if __name__ == '__main__':
    PRINTING = False
    DISABLE_MESSAGE = False
    RENDER = True
    DO_SLEEP = False
    SEARCH = False
    OLD = False
    OLD_2 = False

    SAVE_VIDEO = False

    CHECKPOINT_DIR = '/home/lucius/working/projects/PommermanAgent/scalable_population/agents/pbt_long_burnin_no_spread_population_10_ent_0.15_lr_0.001'
    CHECKPOINT_DIR_2 = '/home/lucius/working/projects/PommermanAgent/scalable_population/agents/pbt_long_burnin_no_spread_population_10_ent_0.15_lr_0.001'

    AGENT = 9
    AGENT_2 = 7

    # Create a set of agents (exactly four)
    agent_list = [
        ScalableAgent(checkpoint_dir=CHECKPOINT_DIR, agent_num=AGENT, printing=PRINTING,
                      disable_message=DISABLE_MESSAGE, old=OLD),
        # ScalableAgent(checkpoint_dir=CHECKPOINT_DIR_2, agent_num=AGENT_2, printing=PRINTING, disable_message=DISABLE_MESSAGE, old=OLD),
        # agents.DockerAgent("scalable_agent", port=10005),
        agents.SimpleAgent(),
        # agents.PlayerAgent(agent_control="arrows"),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=10027),
        # agents.DockerAgent("multiagentlearning/skynet955", port=10003),
        # agents.DockerAgent("multiagentlearning/navocado", port=10051),
        # agents.DockerAgent("multiagentlearning/dypm.1", port=10023),
        # agents.DockerAgent("multiagentlearning/eisenach", port=10021),
        ScalableAgent(checkpoint_dir=CHECKPOINT_DIR, agent_num=AGENT, printing=PRINTING,
                      disable_message=DISABLE_MESSAGE, old=OLD),
        # ScalableAgent(checkpoint_dir=CHECKPOINT_DIR_2, agent_num=AGENT_2, printing=PRINTING, disable_message=DISABLE_MESSAGE, old=OLD),
        # agents.DockerAgent("scalable_agent", port=10006),
        agents.SimpleAgent(),
        # agents.PlayerAgent(agent_control="wasd"),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=10028),
        # agents.DockerAgent('multiagentlearning/skynet955', port=10004),
        # agents.DockerAgent('multiagentlearning/navocado', port=10052),
        # agents.DockerAgent("multiagentlearning/dypm.2", port=10024),
        # agents.DockerAgent("multiagentlearning/eisenach", port=10022),
    ]

    agent_list_2 = [
        agents.SimpleAgent(),
        # agents.PlayerAgent(agent_control="arrows"),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=10029),
        # agents.DockerAgent("multiagentlearning/dypm.1", port=10025),
        # agents.DockerAgent("multiagentlearning/navocado", port=10053),
        # agents.DockerAgent("multiagentlearning/eisenach", port=10023),
        ScalableAgent(checkpoint_dir=CHECKPOINT_DIR, agent_num=AGENT, printing=PRINTING,
                      disable_message=DISABLE_MESSAGE, old=OLD),
        agents.SimpleAgent(),
        # agents.PlayerAgent(agent_control="wasd"),
        # agents.DockerAgent("multiagentlearning/eisenach", port=10024),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=10030),
        # agents.DockerAgent("multiagentlearning/dypm.2", port=10026),
        # agents.DockerAgent("multiagentlearning/navocado", port=10054),
        ScalableAgent(checkpoint_dir=CHECKPOINT_DIR, agent_num=AGENT, printing=PRINTING,
                      disable_message=DISABLE_MESSAGE, old=OLD),
    ]

    # Against skynet955          win percentage: 0.74 tie percentage: 0.26 loss percentage: 0.0
    # Against navocado           win percentage: 0.44 tie percentage: 0.55 loss percentage: 0.01
    # Against dypm.1             win percentage: 0.24 tie percentage: 0.6 loss percentage: 0.16
    # Against eisenach           win percentage: 0.28 tie percentage: 0.1 loss percentage: 0.62
    # Against hakozakijunctions  win percentage: 0.07 tie percentage: 0.41 loss percentage: 0.52

    # saved_agent_9_new
    # eisenach win percentage: 0.38 tie percentage: 0.09 loss percentage: 0.53

    # Make the "Free-For-All" environment using the agent list
    env1 = pommerman.make('PommeRadioCompetition-v2', agent_list)
    env2 = pommerman.make('PommeRadioCompetition-v2', agent_list_2)

    env = env1

    if SAVE_VIDEO:
        save_video(env)
        quit()

    # Run the episodes just like OpenAI Gym
    total_wins = 0
    total_losses = 0
    total_ties = 0
    total_games = 100
    for i_episode in range(total_games):
        env = env1 if i_episode % 2 == 0 else env2

        state = env.reset()
        while True:
            if RENDER:
                env.render(do_sleep=DO_SLEEP)

            if not PRINTING:
                blockPrint()

            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            # time.sleep(0.2)

            if not PRINTING:
                enablePrint()

            if done:
                break

        if i_episode % 2 != 0:
            reward = reward[1:] + [reward[0]]

        print('Episode {} finished. Reward: {}. Other Reward: {}'.format(i_episode, reward[0], reward[1]))

        if reward[0] == 1:
            total_wins += 1
        elif reward[0] == -1 and reward[1] == -1:
            total_ties += 1
        else:
            total_losses += 1

        print('tentative --- win percentage:', total_wins / (i_episode + 1), 'tie percentage:',
              total_ties / (i_episode + 1),
              'loss percentage:', total_losses / (i_episode + 1))

    print('win percentage:', total_wins / total_games, 'tie percentage:', total_ties / total_games, 'loss percentage:',
          total_losses / total_games)
    env.close()
