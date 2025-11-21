from DecisionTree import DecisionTree
from utils import plot_decision_boundary, load_random_dataset, load_circle_dataset
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, dirname, abspath, exists
from os import mkdir

#######################
#### DO NOT MODIFY ####
plots_directory = join(dirname(abspath(__file__)), '..', 'plots')
if not exists(plots_directory):
    mkdir(plots_directory)
########################
########################

def part1():
    X, y = load_random_dataset()
    model = DecisionTree(max_depth=None)
    model.train(X, y)
    accuracy = model.accuracy_score(X, y)

    fig = plot_decision_boundary(X, y, model, title='Part 1: Random Dataset', resolution=0.02)
    fig.show()  # Show plot

    fig.savefig(join(plots_directory, 'part1.png'))
    print('Part 1 Accuracy:', accuracy)


def part2():
    X_train, y_train, X_val, y_val = load_circle_dataset()
    model = DecisionTree(max_depth=None)
    model.train(X_train, y_train)
    training_accuracy = model.accuracy_score(X_train, y_train)
    validation_accuracy = model.accuracy_score(X_val, y_val)

    fig_train = plot_decision_boundary(X_train, y_train, model, title='Part 2: Circle Training')
    fig_val = plot_decision_boundary(X_val, y_val, model, title='Part 2: Circle Validation')
    fig_train.show()
    fig_val.show()

    fig_train.savefig(join(plots_directory, 'part2_training.png'))
    fig_val.savefig(join(plots_directory, 'part2_validation.png'))
    print('Part 2 Training Accuracy:', training_accuracy)
    print('Part 2 Validation Accuracy:', validation_accuracy)


def part3():
    X_train, y_train, X_val, y_val = load_circle_dataset()
    model = DecisionTree(max_depth=1)
    model.train(X_train, y_train)
    training_accuracy = model.accuracy_score(X_train, y_train)
    validation_accuracy = model.accuracy_score(X_val, y_val)

    fig_train = plot_decision_boundary(X_train, y_train, model, title='Part 3: Circle Training (Stump)')
    fig_val = plot_decision_boundary(X_val, y_val, model, title='Part 3: Circle Validation (Stump)')
    fig_train.show()
    fig_val.show()

    fig_train.savefig(join(plots_directory, 'part3_training.png'))
    fig_val.savefig(join(plots_directory, 'part3_validation.png'))
    print('Part 3 Training Accuracy:', training_accuracy)
    print('Part 3 Validation Accuracy:', validation_accuracy)


def part4():
    X_train, y_train, X_val, y_val = load_circle_dataset()
    max_depth_range = range(1, 21)
    training_accuracies = []
    validation_accuracies = []

    for depth in max_depth_range:
        model = DecisionTree(max_depth=depth)
        model.train(X_train, y_train)
        training_accuracies.append(model.accuracy_score(X_train, y_train))
        validation_accuracies.append(model.accuracy_score(X_val, y_val))

    fig = plt.figure()
    plt.plot(max_depth_range, training_accuracies, label='Training Accuracy')
    plt.plot(max_depth_range, validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Part 4: Hyperparameter Tuning')
    plt.legend()
    plt.grid(True)
    plt.show()  # Show plot

    optimal_max_depth = max_depth_range[np.argmax(validation_accuracies)]
    plt.savefig(join(plots_directory, 'part4_hyperparameter_tuning.png'))
    print('Part 4 Optimal Max Depth:', optimal_max_depth)


def part5():
    X_train, y_train, X_val, y_val = load_circle_dataset()
    max_depth_range = range(1, 21)
    validation_accuracies = []

    for depth in max_depth_range:
        model = DecisionTree(max_depth=depth)
        model.train(X_train, y_train)
        validation_accuracies.append(model.accuracy_score(X_val, y_val))
    optimal_max_depth = max_depth_range[np.argmax(validation_accuracies)]

    model = DecisionTree(max_depth=optimal_max_depth)
    model.train(X_train, y_train)
    training_accuracy = model.accuracy_score(X_train, y_train)
    validation_accuracy = model.accuracy_score(X_val, y_val)

    fig_train = plot_decision_boundary(X_train, y_train, model, title='Part 5: Circle Training (Optimal Depth)')
    fig_val = plot_decision_boundary(X_val, y_val, model, title='Part 5: Circle Validation (Optimal Depth)')
    fig_train.show()
    fig_val.show()

    fig_train.savefig(join(plots_directory, 'part5_training.png'))
    fig_val.savefig(join(plots_directory, 'part5_validation.png'))
    print('Part 5 Training Accuracy:', training_accuracy)
    print('Part 5 Validation Accuracy:', validation_accuracy)


if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()
    part5()
