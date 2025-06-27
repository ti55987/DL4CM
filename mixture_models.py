from utils.simulate_utils import (
    action_softmax,
)
import pandas as pd
import numpy as np
import random

class WMMixture:
    # subjective reward for the negative outcome, r0 = 0 reprensents an RL agenet.
    def __init__(self, id, eta2_wm, eta6_wm, r0, phi=0, stickiness=0.0, bias=1, eps=0.0):
        self.id = id
        self.learning_rate = 0
        self.beta = 25  # softmax temperature
        self.stickiness = stickiness
        self.eps = eps
        self.phi = phi
        self.wm_bias = bias
        self.rl_bias = bias
        self.eta2_wm = eta2_wm  # wm weight in policy calculation
        self.eta6_wm = eta6_wm  # wm weight in policy calculation
        self.subjective_reward = [r0, 1]
        self.__Q = {}
        self.__W = {}
        self.num_actions = 0.0
        self.random_action = 0.0
        self.mapping = {}

    # init model should be called for each block
    def init_model(self, learning_rate, stimuli, actions, mapping):
        self.learning_rate = learning_rate
        actions = set(actions)
        stimuli = set(stimuli)
        self.num_actions = len(actions)
        self.mapping = mapping
        self.random_action = 1.0 / self.num_actions
        self.__Q = {}
        for st in stimuli:
            self.__Q[st] = {ac: self.random_action for ac in actions}
            self.__W[st] = {ac: self.random_action for ac in actions}

    def simulate_block(self, num_trials, stimuli, sz=6):
        data = {
            "actions": [],
            "stimuli": [],
            "correct_actions": [],
            "rewards": [],
            "trials": list(range(num_trials)),
            "isswitch": [0] * num_trials,
            "iscorrectaction": [],
            "rpe_Q": [],
        }
        correct_mapping = random.choice(self.mapping)
        previous_action = -1
        for i in range(num_trials):
            s = stimuli[i]
            ac, r = self.select_action(s, correct_mapping[s], sz, previous_action)
            rpe = self.update_values(s, ac, r)
            
            data["stimuli"].append(s)
            data["rpe_Q"].append(rpe)
            data["actions"].append(ac)
            data["rewards"].append(r)
            data["correct_actions"].append(correct_mapping[s])
            data["iscorrectaction"].append(int(ac == correct_mapping[s]))
            previous_action = ac

        data = pd.DataFrame(data)
        data["alpha"] = self.learning_rate
        data["wm_bias"] = self.wm_bias
        data["rl_bias"] = self.rl_bias
        data["eta6_wm"] = self.eta6_wm
        data["beta"] = self.beta
        data["phi"] = self.phi
        data["stickiness"] = self.stickiness
        data["agentid"] = self.id
        data["eps"] = self.eps
        data["eta2_wm"] = self.eta2_wm

        return data
    
    def neg_log_likelihood(self, stimili, actions, rewards, set_size):
        llh = 0
        previous_action = -1
        for st, ac, r in zip(stimili, actions, rewards):
            pi = self.get_policy(st, set_size, previous_action)
            llh += np.log(pi[ac])
            self.update_values(st, ac, r)
            previous_action = ac

        return -llh
    
    def select_action(self, st, correct_action, set_size, previous_action=-1):
        pi = self.get_policy(st, set_size, previous_action)
        ac = np.random.choice(
            self.num_actions, p=pi
        )  # select the action using the probab

        correct = 1 if ac == correct_action else 0

        return ac, correct
    
    def update_values(self, st, ac, reward):
        # Forgetting - fix to case with different Q/W
        for s, action_to_prob in self.__W.items():
            for a in action_to_prob.keys():
                # same thing as WM = WM + forget (1/n - WM)
                self.__W[s][a] = (1.0 - self.phi) * self.__W[s][
                    a
                ] + self.phi * self.random_action

        # Perseveration
        alpha_rl = [self.rl_bias * self.learning_rate, self.learning_rate]
        alpha_wm = [self.wm_bias, 1]

        # Q updates
        # RPE calculation
        rpe = self.subjective_reward[reward] - self.__Q[st][ac]
        self.__Q[st][ac] = self.__Q[st][ac] + alpha_rl[reward] * rpe
        # W updates - one-shot encoding
        self.__W[st][ac] = self.__W[st][ac] + alpha_wm[reward] * (
            reward - self.__W[st][ac]
        )
        return rpe

    def get_policy(self, stimulus, set_size, previous_action=-1):
        Q_st = self.__Q[stimulus].copy()
        W_st = self.__W[stimulus].copy()

        if previous_action != -1:
            Q_st[previous_action] = Q_st[previous_action] + self.stickiness
            W_st[previous_action] = W_st[previous_action] + self.stickiness

        pi_rl = action_softmax(Q_st, self.beta)
        pi_wm = action_softmax(W_st, self.beta)
        
        n_a = len(pi_rl)
        # Mixed WM and RL
        eta_wm = self.eta6_wm if set_size == 6 else self.eta2_wm
        mixed_pi_rl = {
            ac: eta_wm * pi_wm[ac] + (1.0 - eta_wm) * pi_rl[ac] for ac in range(n_a)
        }
        pi = [
            (1.0 - self.eps) * mixed_pi_rl[ac] + self.eps / n_a for ac in range(n_a)
        ]
        return pi
