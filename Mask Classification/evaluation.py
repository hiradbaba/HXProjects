#evaluation phase
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np


class Evaluation():
    def __init__(self, X_test, y_test, y_pred, net):
        self.y_test = y_test
        self.X_test = X_test
        self.y_pred = y_pred
        self.net = net

    def evaluate(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(13,13))
        plot_confusion_matrix(self.net, self.X_test, self.y_test)
        plt.xlabel(classification_report(self.y_test, self.y_pred))
        plt.show()

    def history_stats(self):
        fig, axs = plt.subplots(1, 2)
        fig.set_figheight(10)
        fig.set_figwidth(25)

        train_losses = np.asarray(self.net.history[:, ('train_loss')])
        valid_losses = np.asarray(self.net.history[:, ('valid_loss')])


        plt.setp(axs[0], xticks=np.arange(30), xticklabels=np.arange(1,31),
                yticks=np.arange(10,step=0.2))
        axs[0].set_title('Train Loss vs. Validation Loss')
        plt.setp(axs[1], xticks=np.arange(30), xticklabels=np.arange(1,31),
                yticks=np.arange(110,step=10))
        axs[0].plot(train_losses)
        axs[0].plot(valid_losses)
        axs[0].grid()
        axs[0].set(xlabel='Epochs', ylabel='Loss')


        train_acc = np.asarray(self.net.history[:, ('train_acc')]) * 100
        valid_acc = np.asarray(self.net.history[:, ('valid_acc')]) * 100
        axs[1].plot(train_acc)
        axs[1].plot(valid_acc)
        axs[1].grid()
        axs[1].set_title('Train Accuracy vs Validation Accuracy')
        axs[1].set(xlabel='Epochs', ylabel='Accuracy(\%)')
        plt.show()

class Torch_Evaluator:
    def __init__(self, X_test, y_test, y_pred, net):
        self.y_test = y_test.cpu()
        self.X_test = X_test.cpu()
        self.y_pred = y_pred.cpu()
        self.net = net.cpu()
        self.labels = ['Without Mask','With Mask','Not a Person']
    def evaluate(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        t = cm.max() / 1.5
        fig = plt.figure(figsize=(6,7))
        plt.imshow(cm,cmap=plt.get_cmap('Reds'))
        plt.xticks(np.arange(3),self.labels,rotation=45)
        plt.yticks(np.arange(3),self.labels)
        plt.colorbar()

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j,i,f"{cm[i,j]}", horizontalalignment="center", 
                    color="black" if cm[i,j] < t else "white"
                    )
        print(cm)  
        plt.title(classification_report(self.y_test, self.y_pred))
        plt.subplots_adjust(top=0.775,
                            bottom=0.109,
                            left=0.169,
                            right=0.979)
        fig.tight_layout()
        plt.show()
    
    def evaluate_image(self,image):
        return net(image)









