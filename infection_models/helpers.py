import random
import pandas as pd

def read_file(filename) -> dict:
    data = {}
    with open(filename, 'r+') as text:
        for line in text.readlines():
            key, value = line.strip().split(',')
            data[key] = value

    return data

def round_number(N, percentage) -> int:
    return round((float(percentage)/100.) *N)

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def load_susceptibility_matrix():
    return pd.read_csv('../data/susceptibility_matrix.csv')

def load_recovery_matrix():
    return pd.read_csv('../data/recovery_matrix.csv')

def get_susceptibility_matrix_index(age):
    '''
        The age matrix is 16x16 and it's split in groups of 4,
        We can use whole division to quickly get the index
    '''
    if age >= 75:
        return 15
    else:
        return age//5

def get_recovery_rate(gender, age):
    nodes_recovery = load_recovery_matrix()

    if gender == 0:
        column = 'male'
    elif gender == 1:
        column = 'female'

    if age <= 19:
        return nodes_recovery[column][0]
    elif age >= 60:
        return nodes_recovery[column][5]
    else:
        return nodes_recovery[column][(age//10)-1]

def infection_rate(nodeA, nodeB, G, dataframe):
    ageA = G.nodes[nodeA]['age']
    ageB = G.nodes[nodeB]['age']
    gender = G.nodes[nodeA]['gender']
    ethnicity = G.nodes[nodeA]['ethnicity']

    row = get_susceptibility_matrix_index(ageA)
    col = get_susceptibility_matrix_index(ageB)

    age_infection_rate = dataframe.iloc[row, col]

    # infection probabilities for populations
    gender_infection_rate = [0.17, 0.146] # male, female infection rates
    population_infection_rate =[0.7392, 0.8618, 0.4927, 0.8799] # white, black, mixed, asian

    return age_infection_rate * gender_infection_rate[gender] * population_infection_rate[ethnicity]

def recovery_rate(node,G):
    age = G.nodes[node]['age']
    gender = G.nodes[node]['gender']
    ethnicity = G.nodes[node]['ethnicity']

    population_recover_rate = [0.1585, 0.2910, 0.1923, 0.1585] # white, black, mixed, asian

    gender_recover_rate = get_recovery_rate(gender, age)

    return gender_recover_rate * population_recover_rate[ethnicity]

def infect(active_nodes, G, dataframe):
    for n in active_nodes:
        neighbors = list(G.neighbors(n))
        if len(neighbors) > 0:
            for neighbor in neighbors:
                if G.nodes[n]['status'] == 'S':
                    if random.uniform(0,1) < (infection_rate(n, neighbor, G, dataframe) - 0.076):
                        G.nodes[n]['status'] = 'I'

def recover(active_nodes, G):
    for n in active_nodes:
        if G.nodes[n]['status'] == 'I':
            if random.uniform(0,1) < recovery_rate(n, G):
                G.nodes[n]['status'] = 'S'

def count_compartament_data(G):
    dod = {} # dict of dicts
    for node in G.nodes:
        dod[node] = G.nodes[node]

    df = pd.DataFrame(dod).transpose() # swap rows and columns
    status_counts = df['status'].value_counts()

    return status_counts.S, status_counts.I # return the number of susceptible and infected