
from tkinter import *
from tkinter import filedialog
from tkinter import ttk as t
import os 
import data_process,trainer,evaluator
import pickle 
from tkinter import messagebox
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

'''GUI for testing and evaluating models'''

class GUI:

    def __init__(self,master):
        #creaing root window
        # configuring the intial colors,positions and commands of each componnent
        self.master = master
        self.master = master.configure(background="#1DA1F2")
        self.eval_button = t.Button(master,text="Evaluate",width=13, command=self.eval_command)
        self.predict_button = t.Button(master,text="Predict",width=13, command=self.predict_command)
        self.model_list = Listbox(master,height=10,width=50)
        self.tweet_text  = Text(master, height = 10, width = 40)
        
        self.label = Label(master,text="__Models__",font=("Lucida Console", 11))
        self.tweet_label = Label(master,text="__Tweet__",font=("Lucida Console", 11))
        self.model_list_scorll = Scrollbar(master)
        self.config()
        self.align_components()
        self.model_list_show()
        #the model which is selected by clicking in the model list in GUI
        self.selected_model = None
        #preprocessor to make test set and convert raw text
        self.preproccessor =  data_process.Preprocessor('data/tweet_data.csv')
        X_train,X_test,y_train,y_test = self.preproccessor.create_training_test_set('transform_data')
        self.evaluator = evaluator.Evaluator(X_test,y_test)

        with open('nlp_models/encoder.pkl','rb') as f:
            self.le = pickle.load(f)
        

    #configure conponents colors and events
    def config(self):
        self.label.config(fg="#272727")
        self.label.config(bg="#1DA1F2")
        self.model_list.config(fg="#FFF")
        self.model_list.config(bg="#353535")
        self.tweet_label.config(fg="#272727")
        self.tweet_label.config(bg="#1DA1F2")
        self.model_list.config(yscrollcommand=self.model_list_scorll.set)
        self.model_list.bind("<<ListboxSelect>>",self.get_item)
        self.model_list_scorll.config(command = self.model_list.yview)
    #load model directory into gui
    def model_list_show(self):
        models = os.listdir("./models/")
        for model in models:
            self.model_list.insert(END, model)

    def align_components(self):
        self.eval_button.grid(row=1,column=1,columnspan=3,padx=5,pady=5)
        self.predict_button.grid(row=1,column=7,columnspan=3,padx=5,pady=5)
        self.model_list.grid(row=4,column=1,columnspan=5,padx=1,pady=1)
        self.label.grid(row=3,column=2)
        self.model_list_scorll.grid(row=4,column=6)
        self.tweet_text.grid(row=4,column=7,padx=1,pady=1)
        self.tweet_label.grid(row=3, column=7)
    #change the selected model by clicking
    def get_item(self,event):
        self.selected_model = self.model_list.get(self.model_list.curselection())
    #retrieve input from text box
    def get_input(self):
        return self.tweet_text.get("1.0",END)

    # Classifies the text written in textbox
    def predict_command(self):
        if self.selected_model is not None:
            tweet = self.get_input()
            is_torch = False
            if tweet != '':
                # convert raw tweet to a vector
                tweet_vector = self.preproccessor.infer_vector('d2v',tweet)
                with open(f'models/{self.selected_model}','rb') as f:
                    model = pickle.load(f)
                    if 'torch' in self.selected_model:
                        is_torch = True
                        tweet_vector = torch.tensor(tweet_vector).to(torch.float)
                        if 'conv1d' in self.selected_model:
                            tweet_vector = tweet_vector.reshape(1,1,50)
                        elif 'conv2d' in self.selected_model:
                            tweet_vector = tweet_vector[:-1]
                            tweet_vector = tweet_vector.reshape(1,1,7,7)
                        elif 'rnn' in self.selected_model:
                            tweet_vector = tweet_vector.reshape(1,25,2)
                        elif 'lstm' in self.selected_model:
                            tweet_vector = tweet_vector.reshape(1,25,2)
                        elif 'rcnn' in self.selected_model:
                            tweet_vector = tweet_vector.reshape(1,1,5,10)
                    pred = model.predict(tweet_vector) if is_torch else model.predict([tweet_vector])
                    print(pred)
                    print(pred.dtype)
                    if pred.dtype == 'int64' or pred.dtype == np.int:
                        pred = self.le.inverse_transform(pred)
                    
                    messagebox.showinfo(title='Prediction',message=pred[0])
    #evaluate selected model
    def eval_command(self):
        if 'torch' in self.selected_model:
            if 'conv1d' in self.selected_model:
                self.evaluator.show_evaluate_torch(f'models/{self.selected_model}',(-1,1,50),50,'Conv1D')
            elif 'conv2d' in self.selected_model:
                self.evaluator.show_evaluate_torch(f'models/{self.selected_model}',(-1,1,7,7),49,'Conv2D')
            elif 'rnn' in self.selected_model:
                self.evaluator.show_evaluate_torch(f'models/{self.selected_model}',(-1,25,2),50,"RNN")
            elif 'lstm' in self.selected_model:
                self.evaluator.show_evaluate_torch(f'models/{self.selected_model}',(-1,25,2),50,"LSTM")
            elif 'rcnn' in self.selected_model:
                
                self.evaluator.show_evaluate_torch(f'models/{self.selected_model}',(-1,1,5,10),50,'LSTM-CNN')
        else:   
            self.evaluator.show_evaluate_sklearn(f"models/{self.selected_model}")


if __name__=='__main__':
    root = Tk()
    root.resizable(0,0)
    root.title('G17 NLP Classifier')
    app = GUI(root)
    root.mainloop()