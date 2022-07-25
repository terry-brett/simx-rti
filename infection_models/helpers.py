import pandas as pd
from scipy.spatial import distance

from infection_models import demographic_prediction
from node_info.types import Agent


def read_file(filename) -> dict:
    data = {}
    with open(filename, "r+") as text:
        for line in text.readlines():
            key, value = line.strip().split(",")
            data[key] = value

    return data


def round_number(N, percentage) -> int:
    return round((float(percentage) / 100.0) * N)


def load_susceptibility_matrix():
    return pd.read_csv("../data/susceptibility_matrix.csv")


def get_susceptibility_matrix_index(age):
    """
    The age matrix is 16x16 and it's split in groups of 4,
    We can use whole division to quickly get the index
    """
    if age >= 75:
        return 15
    else:
        return age // 5


def infection_rate(nodeA, nodeB, G):
    dataframe = load_susceptibility_matrix()

    ageA = G.nodes[nodeA]["age"]
    ageB = G.nodes[nodeB]["age"]
    gender = G.nodes[nodeA]["gender"]
    ethnicity = G.nodes[nodeA]["ethnicity"]

    row = get_susceptibility_matrix_index(ageA)
    col = get_susceptibility_matrix_index(ageB)

    age_infection_rate = dataframe.iloc[row, col]

    # infection probabilities for populations
    gender_infection_rate = [0.17, 0.146]  # male, female infection rates
    population_infection_rate = [
        0.7392,
        0.8618,
        0.4927,
        0.8799,
    ]  # white, black, mixed, asian

    return (
        age_infection_rate
        + gender_infection_rate[gender]
        + population_infection_rate[ethnicity]
    ) / 3


def add_to_network(G, agent):
    if agent.id not in G:
        G.add_node(agent.id)
        G.nodes[agent.id]["age"] = agent.age
        G.nodes[agent.id]["gender"] = agent.gender
        G.nodes[agent.id]["ethnicity"] = agent.ethnicity


def connect(frame, x, y, G, faces, uses_camera=False):
    for i in range(len(faces) - 1):
        if not uses_camera:
            agent_a_face_img = frame[
                y : faces[i][1] + faces[i][3], x : faces[i][0] + faces[i][2]
            ]
            agent_b_face_img = frame[
                y : faces[i + 1][1] + faces[i + 1][3],
                x : faces[i + 1][0] + faces[i + 1][2],
            ]
        else:
            agent_a_face_img = faces[0]
            agent_b_face_img = faces[1]

        age_a, gender_a, ethnicity_a = demographic_prediction.predict(agent_a_face_img)
        age_b, gender_b, ethnicity_b = demographic_prediction.predict(agent_b_face_img)

        if (
            age_a is not None
            and age_b is not None
            and gender_a is not None
            and gender_b is not None
            and ethnicity_a is not None
            and ethnicity_b is not None
        ):
            agent_a = Agent(id=i, age=age_a, gender=gender_a, ethnicity=ethnicity_a)
            agent_b = Agent(id=i + 1, age=age_b, gender=gender_b, ethnicity=ethnicity_b)
            add_to_network(G, agent_a)
            add_to_network(G, agent_b)

            if i < len(faces) - 1:
                if not uses_camera:
                    dist = distance.euclidean(faces[i][:2], faces[i + 1][:2])
                else:
                    dist = 120
                if dist < 130:
                    G.add_edge(agent_a.id, agent_b.id)
                    agent_a.infection_rate = infection_rate(agent_a.id, agent_b.id, G)
                    agent_b.infection_rate = infection_rate(agent_b.id, agent_a.id, G)
                    G.nodes[agent_a.id]["infection_rate"] = agent_a.infection_rate
                    G.nodes[agent_b.id]["infection_rate"] = agent_b.infection_rate
