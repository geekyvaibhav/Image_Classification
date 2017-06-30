
import theano
import theano.tensor as T
import numpy as np
import scipy.misc
import matplotlib
import os

matplotlib.use('TkAgg')

TRAIN_PATH_100_10 = 'C:\Users\Vaibhav Sharma\Desktop\Neural_Network_1\cifar_data_100_10\_train'
TEST_PATH_100_10 = 'C:\Users\Vaibhav Sharma\Desktop\Neural_Network_1\cifar_data_100_10\_test'
TRAIN_PATH_1000_100 = 'C:\Users\Vaibhav Sharma\Desktop\Neural_Network_1\cifar_data_1000_100\_train'
TEST_PATH_1000_100 = 'C:\Users\Vaibhav Sharma\Desktop\Neural_Network_1\cifar_data_1000_100\_test'
MIN_WEIGHT = -0.0001
MAX_WEIGHT = 0.001
N_CLASS = 10
N_OUT_NEURONS = N_CLASS
N_EPOCHS = 10
LEARNINING_RATE = 0.01

epoch_wise_loss = []
epoch_wise_error = []
nodes_wise_error = []
train_data_100_10 = np.zeros((3072, 1000))
test_data_100_10 = np.zeros((3072, 100))
train_target_100_10 = np.zeros([N_CLASS, 1000], dtype=np.int64)
test_target_100_10 = np.zeros([N_CLASS, 100], dtype=np.int64)
train_data_1000_100 = np.zeros((3072, 10000))
test_data_1000_100 = np.zeros((3072, 1000))
train_target_1000_100 = np.zeros([N_CLASS, 10000], dtype=np.int64)
test_target_1000_100 = np.zeros([N_CLASS, 1000], dtype=np.int64)


class TheanoTrainParams(object):
    def __init__(self, n_nodes, lambda_value=0.1):
        self.loss_temp = []
        self.predicted_y_index_list = []
        self.actual_y_index_list = []
        self.correct_prediction_count = 0

        self.x = theano.tensor.dvector('x')
        self.y = theano.tensor.lvector('y')

        self.w_hidden_layer, self.w_output_layer = get_weights(n_nodes)
        self.output = theano.tensor.nnet.softmax((theano.tensor.dot(self.w_output_layer, theano.tensor.nnet.
                                                                    relu(
            (theano.tensor.dot(self.w_hidden_layer, self.x) + 1.5),
            alpha=0)) + 1.5).T)
        self.output_sigmoid = theano.tensor.nnet.softmax((theano.tensor.dot(self.w_output_layer, theano.tensor.nnet.
                                                                            sigmoid(
            (theano.tensor.dot(self.w_hidden_layer, self.x) + 1.5))) + 1.5).T)
        self.y_index_predicted = theano.tensor.argmax(self.output[0])
        self.y_index_actual = theano.tensor.argmax(self.y)

        self.cost_unregulated = theano.tensor.sqr(self.output - self.y).sum()
        self.cost_sigmoid = theano.tensor.sqr(self.output_sigmoid - self.y).sum()
        self.neg_likely_hood = theano.tensor.mean(theano.tensor.nnet.categorical_crossentropy(self.output[0], self.y))
        self.cost_regularized = self.neg_likely_hood + lambda_value * (theano.tensor.sqr(self.w_hidden_layer).sum() +
                                                                       theano.tensor.sqr(self.w_output_layer).sum())

        self.regularized_updates = [(self.w_hidden_layer, self.w_hidden_layer -
                                     (LEARNINING_RATE * theano.tensor.grad(self.cost_regularized,
                                                                           self.w_hidden_layer))),
                                    (self.w_output_layer, self.w_output_layer -
                                     (LEARNINING_RATE * theano.tensor.grad(self.cost_regularized,
                                                                           self.w_output_layer)))]

        self.unregulated_updates = [(self.w_hidden_layer, self.w_hidden_layer -
                                     (LEARNINING_RATE * theano.tensor.grad(self.cost_unregulated,
                                                                           self.w_hidden_layer))),
                                    (self.w_output_layer, self.w_output_layer -
                                     (LEARNINING_RATE * theano.tensor.grad(self.cost_unregulated,
                                                                           self.w_output_layer)))]

        self.sigmoid_updates = [(self.w_hidden_layer, self.w_hidden_layer -
                                 (LEARNINING_RATE * theano.tensor.grad(self.cost_sigmoid,
                                                                       self.w_hidden_layer))),
                                (self.w_output_layer, self.w_output_layer -
                                 (LEARNINING_RATE * theano.tensor.grad(self.cost_sigmoid,
                                                                       self.w_output_layer)))]
        self.regularized_grad = theano.function(
            inputs=[self.x, self.y],
            outputs=[self.w_hidden_layer, self.w_output_layer, self.neg_likely_hood,
                     self.y_index_predicted, self.y_index_actual],
            updates=self.regularized_updates)

        self.unregulated_grad = theano.function(
            inputs=[self.x, self.y],
            outputs=[self.w_hidden_layer, self.w_output_layer, self.neg_likely_hood,
                     self.y_index_predicted, self.y_index_actual],
            updates=self.unregulated_updates)

        self.sigmoid_grad = theano.function(
            inputs=[self.x, self.y],
            outputs=[self.w_hidden_layer, self.w_output_layer, self.neg_likely_hood,
                     self.y_index_predicted, self.y_index_actual],
            updates=self.sigmoid_updates)

    def clear_data(self):
        self.loss_temp = []
        self.predicted_y_index_list = []
        self.actual_y_index_list = []
        self.correct_prediction_count = 0


class TheanoTestParams(object):
    def __init__(self, w_hidden_layer, w_output_layer):
        self.predicted_y_index_list = []
        self.actual_y_index_list = []
        self.x = theano.tensor.dvector('x')
        self.y = theano.shared(np.int64(1), 'y')

        self.output = theano.tensor.nnet.softmax((theano.tensor.dot(w_output_layer, theano.tensor.nnet.
                                                                    relu(
            (theano.tensor.dot(w_hidden_layer, self.x) + 1.5),
            alpha=0)) + 1.5).T)
        self.y_index_predicted = theano.tensor.argmax(self.output[0])
        self.updates = [(self.y, self.y_index_predicted)]

        self.f_test = theano.function([self.x], self.y_index_predicted, updates=self.updates)

    def clear_data(self):
        self.predicted_y_index_list = []
        self.actual_y_index_list = []


def get_weights(n_nodes):
    return theano.shared(np.random.uniform(MIN_WEIGHT, MAX_WEIGHT, (n_nodes, 3072)), 'l1_w'), \
           theano.shared(np.random.uniform(MIN_WEIGHT, MAX_WEIGHT, (10, n_nodes)), 'l2_w')


def clear_data():
    global epoch_wise_loss
    global epoch_wise_error
    global nodes_wise_error

    epoch_wise_loss = []
    epoch_wise_error = []
    nodes_wise_error = []


def read_image(file_name):
    return (scipy.misc.imread(file_name).astype(np.float32) / 255).reshape(-1, 1).flatten()


def read_train_data_100_10():
    pos = 0
    for data in sorted(os.listdir(TRAIN_PATH_100_10)):
        train_data_100_10[:, pos] = read_image(os.path.join(TRAIN_PATH_100_10, data))
        train_target_100_10[int(data[:1])][pos] = 1
        pos += 1


def read_test_data_100_10():
    pos = 0
    for data in sorted(os.listdir(TEST_PATH_100_10)):
        test_data_100_10[:, pos] = read_image(os.path.join(TEST_PATH_100_10, data))
        test_target_100_10[int(data[:1])][pos] = 1
        pos += 1


def read_train_data_1000_100():
    pos = 0
    for data in sorted(os.listdir(TRAIN_PATH_1000_100)):
        train_data_1000_100[:, pos] = read_image(os.path.join(TRAIN_PATH_1000_100, data))
        train_target_1000_100[int(data[:1])][pos] = 1
        pos += 1


def read_test_data_1000_100():
    pos = 0
    for data in sorted(os.listdir(TEST_PATH_1000_100)):
        test_data_1000_100[:, pos] = read_image(os.path.join(TEST_PATH_1000_100, data))
        test_target_1000_100[int(data[:1])][pos] = 1
        pos += 1


def train(theano_params, task_name, x_train, y_train, epoch, activation='relu', is_apply_regularization=False, lambda_value=0.1,
          is_loss_required=False,
          is_error_required=False):
    for i in range(x_train.shape[1]):
        if is_apply_regularization:
            train_w_hidden_layer, train_w_output_layer, loss, y_index_predicted, y_index_actual = \
                theano_params.regularized_grad(x_train[:, i], y_train[:, i])
        elif activation == 'sigmoid':
            train_w_hidden_layer, train_w_output_layer, loss, y_index_predicted, y_index_actual = \
                theano_params.sigmoid_grad(x_train[:, i], y_train[:, i])
        else:
            train_w_hidden_layer, train_w_output_layer, loss, y_index_predicted, y_index_actual = \
                theano_params.unregulated_grad(x_train[:, i], y_train[:, i])

        if is_error_required:
            theano_params.predicted_y_index_list.append(y_index_predicted)
            theano_params.actual_y_index_list.append(y_index_actual)
            if y_index_actual == y_index_predicted:
                theano_params.correct_prediction_count += 1

        if is_loss_required:
            theano_params.loss_temp.append(loss)

    if is_loss_required:
        epoch_wise_loss.append(np.mean(theano_params.loss_temp))

    if is_error_required:
        epoch_wise_error.append(float(x_train.shape[1] - theano_params.correct_prediction_count) / x_train.shape[1])

    return train_w_hidden_layer, train_w_output_layer


def test(theanoTestParam, x_test, y_test):
    for i in range(x_test.shape[1]):
        theanoTestParam.predicted_y_index_list.append(theanoTestParam.f_test(x_test[:, i]))
        theanoTestParam.actual_y_index_list.append(np.argmax(y_test[:, i]))

    return theanoTestParam.predicted_y_index_list, theanoTestParam.actual_y_index_list


def generate_confusion_matrix(predicted_y_index_list, actual_y_index_list):
    confusion_matrix = np.zeros((N_CLASS, N_CLASS), dtype=np.int64)
    for i in range(len(predicted_y_index_list)):
        confusion_matrix[predicted_y_index_list[i]][actual_y_index_list[i]] += 1

    return confusion_matrix


def plot_graph(data, x_min, x_max, y_min, y_max, x_label, y_label, title):
    import matplotlib.pyplot as graph
    graph.plot(data)
    graph.xlim(x_min, x_max)
    graph.ylim(y_min, y_max)
    graph.xlabel(x_label)
    graph.ylabel(y_label)
    graph.title(title)
    graph.show()


def exec_task1():
    theano_params = TheanoTrainParams(100)
    for epoch in range(N_EPOCHS):
        theano_params.clear_data()
        w_train_hidden_layer, w_train_output_layer = train(theano_params, 'Task-1', train_data_100_10,
                                                           train_target_100_10, epoch,
                                                           is_loss_required=True,
                                                           is_error_required=True)

    del theano_params

    plot_graph(epoch_wise_loss, 0, N_EPOCHS, np.min(epoch_wise_loss), np.max(epoch_wise_loss), "Epochs",
               "Train Loss", "EPOCHS Vs LOSS")
    plot_graph(epoch_wise_error, 0, N_EPOCHS, 0.0, 1.0, "Epochs",
               "Error rate", "EPOCHS Vs ERROR-RATE")

    theanoTestParam = TheanoTestParams(w_train_hidden_layer, w_train_output_layer)
    predicted_y_index_list, actual_y_index_list = \
        test(theanoTestParam, test_data_100_10, test_target_100_10)

    del theanoTestParam

    conf_matrix = generate_confusion_matrix(predicted_y_index_list, actual_y_index_list)
    print('Confusion matrix for Task-1: ')
    print(conf_matrix)


def exec_task2():
    theano_params = TheanoTrainParams(100)
    for epoch in range(N_EPOCHS):
        theano_params.clear_data()
        w_train_hidden_layer, w_train_output_layer = train(theano_params, 'Task-2', train_data_100_10,
                                                           train_target_100_10, epoch, activation='sigmoid',
                                                           is_loss_required=True,
                                                           is_error_required=True)
    del theano_params
    plot_graph(epoch_wise_loss, 0, N_EPOCHS, np.min(epoch_wise_loss), np.max(epoch_wise_loss), "Epochs",
               "Train Loss", "EPOCHS Vs LOSS")
    plot_graph(epoch_wise_error, 0, N_EPOCHS, 0.0, 1.0, "Epochs",
               "Error rate", "EPOCHS Vs ERROR-RATE")

    theanoTestParam = TheanoTestParams(w_train_hidden_layer, w_train_output_layer)
    predicted_y_index_list, actual_y_index_list = \
        test(theanoTestParam, test_data_100_10, test_target_100_10)

    del theanoTestParam

    conf_matrix = generate_confusion_matrix(predicted_y_index_list, actual_y_index_list)
    print('Confusion matrix for Task-2: ')
    print(conf_matrix)


def exec_task3():
    global epoch_wise_error
    net_nodes = []
    for node_index in range(1, 6):
        epoch_wise_error = []
        net_nodes.append(100 * node_index)
        theano_params = TheanoTrainParams(100 * node_index)
        for epoch in range(N_EPOCHS):
            theano_params.clear_data()
            train(theano_params, 'Task-3', train_data_100_10, train_target_100_10, epoch,
                  is_error_required=True)
        nodes_wise_error.append(np.mean(epoch_wise_error))
        del theano_params

    import matplotlib.pyplot as graph
    graph.xlabel("Neurons")
    graph.ylabel("Error rate")
    graph.title("ERROR-RATE Vs NUMBER-OF-NEURONS")
    graph.ylim(0.0, 1.0)
    graph.xlim(100, 500)
    graph.plot(net_nodes, nodes_wise_error)
    graph.show()


def exec_task4():
    for lambda_value in [0.1, 0.2, 0.3, 0.4, 0.5]:
        theano_params = TheanoTrainParams(500, lambda_value)
        for epoch in range(N_EPOCHS):
            theano_params.clear_data()
            w_train_hidden_layer, w_train_output_layer = train(theano_params, 'Task-4', train_target_100_10,
                                                               train_target_100_10, epoch,
                                                               is_apply_regularization=True, lambda_value=lambda_value)
        del theano_params

        theanoTestParam = TheanoTestParams(w_train_hidden_layer, w_train_output_layer)
        predicted_y_index_list, actual_y_index_list = \
            test(theanoTestParam, test_data_100_10, test_target_100_10)

        del theanoTestParam

        conf_matrix = generate_confusion_matrix(predicted_y_index_list, actual_y_index_list)
        print('Task-4: Confusion matrix for lambda: %f' % lambda_value)
        print(conf_matrix)


def exec_task5():
    read_train_data_1000_100()
    read_test_data_1000_100()
    best_arch_n_nodes = 500
    best_arch_lambda = 0.1

    theano_params = TheanoTrainParams(best_arch_n_nodes, best_arch_lambda)
    for epoch in range(N_EPOCHS):
        w_train_hidden_layer, w_train_output_layer = train(theano_params, 'Task-5', train_data_1000_100,
                                                           train_target_1000_100,
                                                           epoch, is_apply_regularization=True,
                                                           is_error_required=True, is_loss_required=True,
                                                           lambda_value=best_arch_lambda)

    del theano_params

    plot_graph(epoch_wise_loss, 0, N_EPOCHS, np.min(epoch_wise_loss), np.max(epoch_wise_loss), "Epochs",
               "Train Loss", "EPOCHS Vs LOSS")
    plot_graph(epoch_wise_error, 0, N_EPOCHS, 0.0, 1.0, "Epochs",
               "Error rate", "EPOCHS Vs ERROR-RATE")

    theanoTestParam = TheanoTestParams(w_train_hidden_layer, w_train_output_layer)
    predicted_y_index_list, actual_y_index_list = \
        test(theanoTestParam, test_data_1000_100, test_target_1000_100)

    del theanoTestParam

    conf_matrix = generate_confusion_matrix(predicted_y_index_list, actual_y_index_list)
    print('Task-5: Confusion matrix for nodes(%d) & lambda(%f)' % (best_arch_n_nodes, best_arch_lambda))
    print(conf_matrix)


def execute_tasks():
    exec_task1()
    clear_data()
    exec_task2()
    clear_data()
    exec_task3()
    clear_data()
    exec_task4()
    clear_data()
    exec_task5()


def read_dataset():
    read_train_data_100_10()
    read_test_data_100_10()


if __name__ == "__main__":
    read_dataset()
    execute_tasks()
