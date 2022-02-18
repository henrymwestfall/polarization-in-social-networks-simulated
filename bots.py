import math
from agent import Agent
from message import Message


class Bot(Agent):
    """
    Bots aim to push a partisan agenda by leaving likes on messages that match
    their ideology and by posting consistently partisan messages.
    """
    def __init__(self, sim, node_id: int, ideology: 'list[int]'):
        super().__init__(sim, node_id)
        self.is_bot = True
        self.ideology = ideology

    
    def read_feed(self) -> None:
        for msg in self.feed:
            # leave a like if ideology matches, regardless of magnitude
            if math.copysign(1, msg.magnitude) == self.ideology[msg.issue]:
                msg.author.receive_like(msg.issue)

            # increase confidence in issue
            self.increase_confidence_in_issue(
                msg.issue,
                msg.magnitude,
                1.0
            )
        self.express_belief_state()


    def get_magnitude(self, issue: int) -> int:
        sign = self.ideology[issue]
        neighborhood_state = max(
            self.belief_state_confidences[issue].keys(), 
            key=lambda v: self.belief_state_confidences[issue][v]
        )
        return neighborhood_state + sign


    def express_belief_state(self) -> None:
        for issue in range(len(self.belief_state_confidences)):
            self.next_expressed_belief_state[issue] = self.get_magnitude(issue)
        self.belief_state_history.append(self.next_expressed_belief_state)


    