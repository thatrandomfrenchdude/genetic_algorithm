# references
# https://medium.com/sigmoid/https-medium-com-rishabh-anand-on-the-origin-of-genetic-algorithms-fc927d2e11e0
# genetic algorithm crossover strategies - https://www.ripublication.com/ijcir17/ijcirv13n7_15.pdf
# https://medium.com/xrpractices/reinforcement-learning-vs-genetic-algorithm-ai-for-simulations-f1f484969c56

# two functions should be implemented before learning:
# 1. fitness function
# 2. mutation function

# what is the stopping criteria?
# can be a minimum threshold of fitness

# mini project
# design a genetic algorithm to build the best mnist classifier in pytorch

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import random

def serve_model(epochs, units1, act1, units2, act2, classes, act3, loss, opt, xtrain, ytrain, summary=False):
    model = Sequential()
    model.add(Dense(units1, input_shape=[784,]))
    model.add(Activation(act1))
    model.add(Dense(units2))
    model.add(Activation(act2))
    model.add(Dense(classes))
    model.add(Activation(act3))
    model.compile(loss=loss, optimizer=opt, metrics=['acc'])
    if summary:
        model.summary()

    model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, verbose=0)

    return model

def fitness(networks):
    for network in networks:
        hyperparams = network.init_hyperparams()
        epochs = hyperparams['epochs']
        units1 = hyperparams['units1']
        act1 = hyperparams['act1']
        units2 = hyperparams['units2']
        act2 = hyperparams['act2']
        act3 = hyperparams['act3']
        loss = hyperparams['loss']
        opt = hyperparams['optimizer']

        try:
            model = serve_model(epochs, units1, act1, units2, act2, classes, act3, loss, opt, x_train, y_train)
            accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
            network._accuracy = accuracy
            print ('Accuracy: {}'.format(network._accuracy))
        except:
            network._accuracy = 0
            print ('Build failed.')

    return networks

def selection(networks):
    networks = sorted(networks, key=lambda network: network._accuracy, reverse=True)
    networks = networks[:int(0.2 * len(networks))]

    return networks

def crossover(networks):
    offspring = []
    for _ in range(int((population - len(networks)) / 2)):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)
        child1 = Network()
        child2 = Network()

        # Crossing over parent hyper-params
        child1._epochs = int(parent1._epochs/4) + int(parent2._epochs/2)
        child2._epochs = int(parent1._epochs/2) + int(parent2._epochs/4)

        child1._units1 = int(parent1._units1/4) + int(parent2._units1/2)
        child2._units1 = int(parent1._units1/2) + int(parent2._units1/4)

        child1._units2 = int(parent1._units2/4) + int(parent2._units2/2)
        child2._units2 = int(parent1._units2/2) + int(parent2._units2/4)

        child1._act1 = parent2._act2
        child2._act1 = parent1._act2

        child1._act2 = parent2._act1
        child2._act2 = parent1._act1

        child1._act3 = parent2._act2
        child2._act3 = parent1._act2

        offspring.append(child1)
        offspring.append(child2)

    networks.extend(offspring)

    return networks

def mutate(networks):
    for network in networks:
        if np.random.uniform(0, 1) <= 0.1:
            network._epochs += np.random.randint(0,100)
            network._units1 += np.random.randint(0,100)
            network._units2 += np.random.randint(0,100)

    return networks

