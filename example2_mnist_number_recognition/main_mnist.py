from perceptron_simple import PerceptronSimpleTrainer


def main():
    train_data_path = "data/mnist_train.csv"
    test_data_path = "data/mnist_test.csv"
    perceptron_simple_trainer = PerceptronSimpleTrainer(
        train_data_path,
        test_data_path,
        learn_rate=0.01,
     )
    perceptron_simple_trainer.train()


if __name__ == "__main__":
    main()
