from utils.simulate_utils import choose_different_array, action_softmax, generate_shuffled_list
import pandas as pd
import numpy as np
import random

class PRL:
    def __init__(
        self,
        beta,
        pval,
        id,
        phi=0,
        stickiness=0.0,
    ):
        self.alpha = 0
        self.beta = beta  # softmax temperature
        self.phi = phi
        self.pval = pval
        self.id = id
        self.__Q = {}
        self.mapping = {}
        self.stickiness = stickiness

    def init_model(self, alpha, stimuli, actions, mapping):
        self.alpha = alpha
        actions = set(actions)
        stimuli = set(stimuli)
        self.num_actions = len(actions)
        self.mapping = mapping
        self.random_action = 1.0 / self.num_actions
        self.__Q = {}
        for st in stimuli:
            self.__Q[st] = {ac: self.random_action for ac in actions}


    def simulate_block(self, num_trials, stimuli, min_switch):
        data = {
            "agentid" : [self.id] * num_trials,
            'actions': [],
            'stimuli': [],
            'correct_actions' : [],
            'rewards': [],
            'trials': list(range(num_trials)) ,
            'isswitch':[0] * num_trials,
            'iscorrectaction': [],
            'alpha': [self.alpha]*num_trials,
            'beta': [self.beta]*num_trials,
            'phi': [self.phi]*num_trials,
            'stickiness': [self.stickiness]*num_trials,
            'rpe_history': [],
            'unchosen_rpe_history': [],
        }
        correct_mapping = random.choice(self.mapping)
        # possible_stimuli = list(self.__Q.keys())
        # num_iter = int(num_trials/len(possible_stimuli))
        #stimuli = create_stim_sequence(num_iter, len(possible_stimuli))
        #stimuli = generate_shuffled_list(num_trials, possible_stimuli)
        # the number of correct trials required for the correct action to switch
        currLength = min_switch+random.randint(0, 5)
        currCum = 0 # initialize cumulative reward
        previous_action = -1
        for i in range(num_trials):
            s = stimuli[i]
            ac, r = self.select_action(s, correct_mapping[s], previous_action)
            rpe, rpe_unchosen = self.update_values(s, ac, r)

            data['stimuli'].append(s)
            data['rpe_history'].append(rpe)
            data['unchosen_rpe_history'].append(rpe_unchosen)
            data['actions'].append(ac)
            data['rewards'].append(r)
            data['correct_actions'].append(correct_mapping[s])
            data['iscorrectaction'].append(int(ac == correct_mapping[s]))
            previous_action = ac
            currCum = currCum+r  # update cumulative reward
            # check for the counter of the trials required to switch correct actions
            if (r==1) and (currCum>=currLength):
              correct_mapping = choose_different_array(self.mapping, correct_mapping)
              currLength = min_switch+random.randint(0,5)
              currCum=0
              if i < num_trials-1:
                data['isswitch'][i+1]=1

        return pd.DataFrame(data)

    def select_action(self, st, correct_action, previous_action=-1):
        pi = self.get_policy(st, previous_action)
        ac = np.random.choice(
            self.num_actions, p=pi
        )  # select the action using the probab

        correct = 1 if ac == correct_action else 0

        ran_val = random.random()
        if ran_val<self.pval: # reward with p probability
          reward = correct
        else:
          reward = 1-correct

        return ac, reward

    def update_values(self, st, ac, reward):
        # Forgetting - fix to case with different Q/W
        for s, action_to_prob in self.__Q.items():
            for a in action_to_prob.keys():
                # same thing as WM = WM + forget (1/n - WM)
                self.__Q[s][a] = (1.0 - self.phi) * self.__Q[s][
                    a
                ] + self.phi * self.random_action

        rpe = reward - self.__Q[st][ac]
        # Q updates
        self.__Q[st][ac] = self.__Q[st][ac] + self.alpha * rpe
        # if not self.enabled_counterfactual_learning:
        #   return rpe, 0

        # action that's not selected (for counterfactual learning)
        for x in list(np.arange(self.num_actions)):
          if x == ac:
            continue

          rpe_unchosen = (1-reward)-self.__Q[st][x] #RPE for the unselected action
          self.__Q[st][x] += (self.alpha*rpe_unchosen)

        return rpe, rpe_unchosen

    def get_policy(self, stimulus, previous_action=-1):
        Q_st = self.__Q[stimulus].copy()
        if previous_action != -1:
            Q_st[previous_action] = Q_st[previous_action] + self.stickiness        

        return action_softmax(Q_st, self.beta)
    
