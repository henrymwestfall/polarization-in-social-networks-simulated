The human trials highlighted several key shortcomings of my simulation model:
1. The lack of nuance in belief states on individual issues made the choices fairly arbitrary due to how quickly these states converged. Participants did not gradually become more confident in their belief on an issue; rather, randomness at initialization made participants think that their social networks were already polarized, when in fact they were not. This demonstrates the power of framing an issue as binary, but is not very useful for analyzing other causes of polarization.
2. Any community or sense of relationship with other participants was difficult to observe. While the simulation did store an `r` value that measured the participants relationships with others, participants rarely actually remembered or even noticed the (randomized) identities behind these relationships. In other words, the simulation fully dehumanized communication. This is somewhat consistent with social media; however, this is not consistent with general social networks.
3. Either edges between participants on the social network graph must be defined more definitively, or the edge weighting and pagerank parameters must be tuned further. Regardless of the initial topology (random, small-world, clique based, etc.), the graphs behaved effectively like egalitarian networks. Thus the human experiments and current simulation model support Centola's work, but do not provide much new insight.

Here I seek to create a fully simulated model to address these three main issues without changing the essence of the simulation itself.
1. *Lack of nuance*: Ideology will still be divided into a number of issues; however, an agent's opinion on an issue will not be binary. Rather, agents will select a belief state on the interval [-5, 5]. This has several implications for the "like" mechanisms. Before, a "like" occured with exact synchrony. Now, likes will occur when two belief states on a given issue are within ±1 of each other. Furthermore, when randomizing initial belief states, agents will select a random state on this interval, not a random binary state.
2. *Community and sense of relationships*: This is a difficult issue to solve for human participants, but is simpler to simulate; thus, I will focus on the fully simulated experiments and relegate this issue in human experiments to future studies. Agents will store their confidence in their neighbors using Bayes theorem (?). Confidence shall increase under the same circumstances that a like would occur.
3. *Edges*: I discuss this below.

To simplify the number of parameters to tune, I will also adjust how rewiring works. This is the largest departure from the original simulation, but I think it is necessary considering time constraints. Before, edges were "rewired" based on `r` values, local pagerank, and cosine similarity. Now, I think I will simply evolve the graph with definite connections. Rewiring will be tied to confidence: when confidence in a connection drops below 0.2, that connection will be dropped. When confidence in a connection exceeds 0.8, there will be a chance for an agent to form a new connection with a neighbor of the esteemed connectee (I explained this poorly here; please refer to me or the code).

At each iteration, a number of connections based on feed size will be randomly selected to form a feed.


**What Belief States Are**
For each issue, Agents stores 

**How an Agent reads a message**
Agents read messages one by one, but update their belief state confidences simultaneously using the same current-next system as before. Each message consists of a belief state expression about ONE issue. 