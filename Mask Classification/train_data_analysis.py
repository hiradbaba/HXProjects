import matplotlib.pyplot as plt
import pickle
import numpy as np

def plotter(training_data):
    for id, td in enumerate(training_data, start=1):
        array_size = len(np.array(td.train_accuracy_avg))
        plt.figure()
        plt.subplot(1,2,1)
        plt.title("Train vs Validation Acc.")
        plt.plot(np.arange(1,array_size+1),np.array(td.train_accuracy_avg), label="Train Acc.")
        plt.plot(np.arange(1,array_size+1),np.array(td.validation_accuracy_avg), label="Validation Acc.")
        plt.legend()
        plt.subplot(1,2,2)
        plt.title("Train vs Validation Loss")
        plt.plot(np.arange(1,array_size+1),np.array(td.train_loss_avg), label="Train Loss")
        plt.plot(np.arange(1,array_size+1),np.array(td.validation_loss_avg), label="Validation Loss")
        plt.legend()
        plt.savefig(f'plots/first_plot_fold_{id}.png')

with open("train_data/10_fold_training_data_fold.pkl", "rb") as f:
    train_data = pickle.load(f)
    plotter(train_data)