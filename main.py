# from keras.datasets import mnist
# from keras.utils import to_categorical

from genetic import fitness, selection, crossover, mutate
from network import init_networks

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(60000, 784).astype('float32')[:1000] / 255
# x_test = x_test.reshape(10000, 784).astype('float32')[:1000] / 255
# y_train = to_categorical(y_train, 10)[:1000]
# y_test = to_categorical(y_test, 10)[:1000]

classes = 10
batch_size = 64
population = 20
generations = 100
threshold = 0.995

def main():
    networks = init_networks(population)

    for gen in range(generations):
        print ('Generation {}'.format(gen+1))

        networks = fitness(networks)
        networks = selection(networks)
        networks = crossover(networks)
        networks = mutate(networks)

        for network in networks:
            if network._accuracy > threshold:
                print ('Threshold met')
                print (network.init_hyperparams())
                print ('Best accuracy: {}'.format(network._accuracy))
                exit(0)

if __name__ == '__main__':
    main()