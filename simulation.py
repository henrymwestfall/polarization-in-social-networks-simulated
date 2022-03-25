import json
import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from typing import Iterator

from network import Network
from agent import Agent
from bots import Bot


class Simulation:
    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "gold",
        "saddlebrown",
        "fuchsia",
        "cyan",
        "lime"
    ]

    def __init__(self, params):
        self.params = params
        self.seed = self.params["seed"]
        self.name = self.params["name"]
        self.feed_size = self.params["feed_size"]
        self.num_issues = self.params["num_issues"]
        self.state_space_depth = self.params["state_space_depth"]
        self.state_space = range(-self.state_space_depth, self.state_space_depth + 1)

        self.rng = np.random.default_rng(self.seed)
        self.network = Network(self.rng, self.params)

        ideology_1 = [0 for _ in self.state_space]
        ideology_2 = [0 for _ in self.state_space]
        ideology_1[0] = -1
        ideology_2[0] = 1
        num_bots = self.params["partisans"]
        self.agents = []
        for i, u in enumerate(self.network.get_shuffled_nodes()):
            if i < num_bots / 2:
                self.agents.append(Bot(self, u, ideology_1))
            elif i < num_bots:
                self.agents.append(Bot(self, u, ideology_2))
            else:
                self.agents.append(Agent(self, u))

        self.current_step = 0

        self.labels_by_step = []
        self.silhouettes_by_step = []
        self.polarization_over_time = []


    def send_all_messages(self) -> None:
        self.network.reset_communication_graph()
        for agent in self.agents:
            possible_influencers = list(self.network.get_influencers(agent.node_id))
            num_possible_influencers = len(possible_influencers)
            sample_size = min(self.feed_size, num_possible_influencers)
            influencers = self.rng.choice(
                possible_influencers, 
                size=sample_size,
                replace=False
            )

            for infl_id in influencers:
                infl = self.agents[infl_id]
                agent.feed.append(infl.get_message())

                self.network.log_communication(agent.node_id, infl_id)

            agent.read_feed()


    def run(self, length: int = None) -> None:
        try:
            if length == None:
                length = self.params["simulation_length"]

            tenth = length // 10
            hundredth = length // 100
            print(f"Running simulation for {length} steps")
            pos = None
            for i in range(length):
                self.current_step = i
                self.step()
                if i % tenth == 0:
                    print(f"{10 * i // tenth}%", end="", flush=True)
                elif i % hundredth == 0:
                    print(".", end="", flush=True)

                model, sil  = self.calculate_belief_state_kmeans()
                self.labels_by_step.append(len(model.cluster_centers_))
                self.silhouettes_by_step.append(sil)
                self.polarization_over_time.append(self.calculate_polarization(model.cluster_centers_))
            print("100%")
        except KeyboardInterrupt:
            print(f"\nKeyboard Interrupt, stopping simulation at step {self.current_step}")
        
        with open(self.name + "_silhouettes.txt", "w") as f:
            for sil in self.silhouettes_by_step:
                f.write(str(sil) + "\n")
        with open(self.name + "_polarization.txt", "w") as f:
            for pol in self.polarization_over_time:
                f.write(str(pol) + "\n")
    

    def step(self) -> None:
        self.send_all_messages()
        for agent in self.agents:
            agent.sync_belief_state()

    
    def get_random_issue(self) -> int:
        return self.rng.integers(low=0, high=self.num_issues, size=1)[0].item()


    def get_blank_belief_state(self) -> 'list[int]':
        return [0 for _ in range(self.num_issues)]


    def sever_edge(self, agent_a: Agent, agent_b: Agent) -> None:
        self.network.sever_connection(agent_a.node_id, agent_b.node_id)


    def relocate(self, agent: Agent) -> None:
        self.network.relocate(agent.node_id, self.params["node_out_degree"])


    def clique_of_agent(self, agent: Agent) -> Iterator[Agent]:
        clique = self.network.node_cliques[agent.node_id]
        for node_id in clique:
            yield self.agents[node_id]


    def form_new_edge_by_association(self, agent: Agent, influencer: Agent) -> None:
        # form a new edge with an influencer of an influencer of the agent
        first_influencers = self.network.get_influencers(agent.node_id)
        second_influencers = [
                node 
                for node in self.network.get_influencers(influencer.node_id) 
                if not (node in first_influencers \
                    or self.network.has_been_severed(agent.node_id, node))
            ]
        if len(second_influencers) > 0:
            new_v = self.rng.choice(second_influencers)
            self.network.graph.add_edge(agent.node_id, new_v)


    def get_colors_by_kmeans(self, binary=False, plot=True) -> list:
        labels = self.calculate_belief_state_kmeans(binary, plot)[0].labels_
        return [self.colors[l] for l in labels]

    
    def get_edge_colors_by_trust(self) -> list:
        return [
            [0, 0, 0, self.agents[u].get_confidence_p_in_agent(self.agents[v])]
            for u, v in self.network.graph.edges
        ]


    # Analysis methods
    def calculate_polarization(self, cluster_centers):
        if len(cluster_centers) == 2:
            a, b = cluster_centers
            return np.linalg.norm(a-b) / np.linalg.norm(
                np.full(self.num_issues, 2 * self.state_space_depth + 1)
            )
        return 0


    def calculate_belief_state_kmeans(self, binary=False, plot=False):
        X = np.array([agent.get_average_belief_states(binary=binary) for agent in self.agents if isinstance(agent, Agent)])

        models = []
        inertias = []
        K = range(1, 10)

        for k in K:
            models.append((model := KMeans(n_clusters=k, random_state=0)))
            model.fit(X)
            inertias.append(model.inertia_)

        silhouette_scores = [0]
        for model in models[1:]:
            silhouette_scores.append(
                silhouette_score(X, model.labels_, metric="euclidean"))
        high_score = max(silhouette_scores)
        index = silhouette_scores.index(high_score)
        best = models[index]

        d = None
        if index == 1:
            a, b = best.cluster_centers_
            d = np.linalg.norm(a-b) / np.linalg.norm(
                np.full(self.num_issues, 2 * self.state_space_depth + 1)
            )
        
        if plot:
            plt.title(f"Average Silhouette Coefficients for {self.name}")
            plt.ylabel(f"Average Silhouette Score")
            plt.xlabel(f"Number of Clusters")
            plt.ylim([0, 1])
            plt.plot(K, silhouette_scores)
            plt.axvline(x=index + 1, linestyle='--')
            plt.show()

            print(best.cluster_centers_)
            print(d)
            

        return best, high_score


if __name__ == "__main__":
    with open("parameters.json") as f:
        params = json.load(f)

    sim = Simulation(params)
    sim.run()

    plt.title("Labels vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Labels")
    plt.plot(range(sim.current_step + 1), sim.labels_by_step)
    plt.show()
    print(sim.labels_by_step)

    plt.title("Silhouette vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Silhouette Score")
    plt.plot(range(sim.current_step + 1), sim.silhouettes_by_step)
    plt.show()
    print(sim.silhouettes_by_step)

    plt.title(f"Polarization vs Iteration for {sim.params['partisans']} Partisan Bots")
    plt.xlabel("Iteration")
    plt.ylabel("Polarization")
    plt.plot(range(sim.current_step + 1), sim.polarization_over_time)
    plt.show()
    print(sim.polarization_over_time)

    sim.network.display_connection_graph(
        "Network",
        sim.get_colors_by_kmeans(False),
        sim.get_edge_colors_by_trust(),
    )