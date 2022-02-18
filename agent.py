import math
import matplotlib.pyplot as plt
import numpy as np

from message import Message


class Agent:
    def __init__(self, sim, node_id: int):
        self.sim = sim
        self.node_id = node_id
        self.is_bot = False

        # load behavior parameters
        self.initial_confidence_numer = self.sim.params["initial_confidence_numer"]
        self.initial_confidence_denom = self.sim.params["initial_confidence_denom"]
        self.disconnect_confidence_thresh = self.sim.params["disconnect_confidence_thresh"]
        self.connect_confidence_thresh = self.sim.params["connect_confidence_thresh"]
        self.update_edges_probability = self.sim.params["update_edges_probability"]
        self.like_weight = self.sim.params["like_weight"]
        self.resonance_like_thresh = self.sim.params["resonance_like_thresh"]
        self.belief_state_decay = self.sim.params["belief_state_decay"]

        self.connection_confidences: 'dict[Agent, float]' = {}
        
        # ({state: confidence magnitudes})
        # TODO: change this
        self.belief_state_confidences: 'tuple[dict[int, float], ...]' = \
            tuple({
                state: 0.01 for state in self.sim.state_space
            } for _ in range(self.sim.num_issues)
        )
        # for i in range(self.sim.num_issues):
        #     self.set_belief_state_confidence_value(
        #         i, 
        #         self.sim.rng.choice(self.sim.state_space), 
        #         1
        #     )

        self.belief_state_history = []
        self.expressed_belief_state = self.sim.get_blank_belief_state()
        self.next_expressed_belief_state = self.expressed_belief_state.copy()

        self.feed: 'list[Message]' = []


    def get_message(self) -> Message:
        return Message(
            self, 
            (issue := self.sim.get_random_issue()),
            self.get_magnitude(issue)
        )


    def get_magnitude(self, issue):
        stances = []
        ws = []
        for stance, w in self.belief_state_confidences[issue].items():
            stances.append(stance)
            ws.append(w)
        sum_w = sum(ws)
        if sum_w > 0:
            p = [w / sum_w for w in ws]
        else:
            p = [1 / len(ws) for _ in ws]
        magnitude = self.sim.rng.choice(stances, p=p)
        return magnitude


    def read_feed(self) -> None:
        for msg in self.feed:
            # calculate base confidence in message
            numer, denom = self.get_confidence_in_agent(msg.author)
            confidence_p = numer / denom
            
            # if message resonates, increase confidence in message and author
            # otherwise, don't
            if self.sim.current_step > 0:
                resonates = self.get_message_resonance(msg)
                if resonates:
                    self.connection_confidences[msg.author] = (numer + 1, denom + 2)
                    msg.author.receive_like(msg.issue)
                else:
                    self.connection_confidences[msg.author] = (numer, denom + 2)

            # regardless, increase confidence in issue
            self.increase_confidence_in_issue(
                    msg.issue, 
                    msg.magnitude, 
                    confidence_p
            )

            # if trust in author is high enough, potentially form connection
            if confidence_p >= self.connect_confidence_thresh \
                    and self.sim.rng.random() < self.update_edges_probability:
                self.sim.form_new_edge_by_association(self, msg.author)

        self.feed.clear()
        self.express_belief_state()
        self.decay_belief_state_confidences()

        # potentially move
        if self.check_edges_for_move():
            self.sim.relocate(self)


    def decay_belief_state_confidences(self) -> None:
        for confidences in self.belief_state_confidences:
            for state in confidences.keys():
                confidences[state] *= self.belief_state_decay


    def check_edges_for_move(self) -> bool:
        """
        Return whether this Agent should move to a different clique.
        """
        ps = []
        for n in self.sim.clique_of_agent(self):
            p = self.get_confidence_p_in_agent(n)
            ps.append(p)
        avg_p = sum(ps) / len(ps)
        return avg_p < self.disconnect_confidence_thresh


    def receive_like(self, issue: int):
        self.increase_confidence_in_issue(
            issue, self.expressed_belief_state[issue], self.like_weight
        )


    def express_belief_state(self) -> None:
        for issue, confidences in enumerate(self.belief_state_confidences):
            opts = []
            ws = []
            for state, confidence in confidences.items():
                opts.append(state)
                ws.append(confidence)
            sum_w = sum(ws)
            p = [w / sum_w for w in ws]
            self.next_expressed_belief_state[issue] = self.sim.rng.choice(opts, p=p)
        self.belief_state_history.append(self.next_expressed_belief_state)

    
    def set_belief_state_confidence_value(self, i: int, m: int, c: float) -> None:
        self.belief_state_confidences[i][m] = c


    def sync_belief_state(self) -> None:
        self.expressed_belief_state = self.next_expressed_belief_state
        self.next_expressed_belief_state = self.sim.get_blank_belief_state()


    def get_message_resonance(self, msg: Message, tail=5) -> bool:
        if self.sim.current_step > 0:
            avgs = self.get_average_belief_states(tail)
            avg = avgs[msg.issue]
            if avg != 0:
                same_side = msg.magnitude / avg > 0
            else:
                same_side = True
            return same_side and abs(avg) + 1 >= abs(msg.magnitude)
        return True


    def increase_confidence_in_issue(self, issue: int, magnitude: int, confidence: float) -> None:
        if not magnitude in self.belief_state_confidences[issue]:
            self.belief_state_confidences[issue][magnitude] = 1
        self.belief_state_confidences[issue][magnitude] += confidence


    def get_confidence_in_agent(self, agent: 'Agent') -> 'tuple[int, int]':
        if not agent in self.connection_confidences:
            self.connection_confidences[agent] = \
                (self.initial_confidence_numer, self.initial_confidence_denom) # e.g. 1 / 2
        return self.connection_confidences[agent]

    
    def get_confidence_p_in_agent(self, agent: 'Agent') -> float:
        numer, denom = self.get_confidence_in_agent(agent)
        return numer / denom


    def plot_belief_state_evolution(self, tail=10) -> None:
        history_matrix = np.array(self.belief_state_history[-tail:])

        xs = range(history_matrix.shape[0])

        for i, color in zip(range(history_matrix.shape[1]), self.sim.colors):
            ys = history_matrix[:, i]
            plt.plot(xs, ys, color=color, alpha=0.2)

            avg = ys.sum() / ys.shape[0]
            ys = [avg for _ in xs]
            plt.plot(xs, ys, color=color)
        plt.show()

    
    def get_average_belief_states(self, tail=10, binary=False) -> 'list[int]':
        history_matrix = np.array(self.belief_state_history[-tail:])
        avgs = []
        for i in range(history_matrix.shape[1]):
            ys = history_matrix[:, i]
            avg = ys.sum() / ys.shape[0]

            if binary:
                avgs.append(math.copysign(1.0, avg))
            else:
                avgs.append(avg)
        return avgs        