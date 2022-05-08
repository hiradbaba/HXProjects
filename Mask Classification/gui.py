from threading import Thread
from tkinter import *
from tkinter import filedialog
from tkinter import ttk as t
from datetime import datetime
from time import sleep
import os,cv2
import torch
from evaluation import Torch_Evaluator
from preprocess import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from PIL import Image,ImageTk
from torch.autograd import Variable
class GUI:

    def __init__(self,master):
        self.master = master
        self.master = master.configure(background="#272727")
        self.eval_button = t.Button(master,text="Evaluate",width=13, command=lambda: self.eval_command("./csv/test_fmn.csv"))
        self.fm_eval_button = t.Button(master,text="FM Eval",width=13, command=lambda: self.eval_command("./csv/test_fm.csv"))
        self.female_eval_button = t.Button(master,text="Female Eval",width=13, command=lambda: self.eval_command("./csv/test_f.csv"))
        self.male_eval_button = t.Button(master,text="Male Eval",width=13, command=lambda: self.eval_command("./csv/test_m.csv"))
        self.camera_button = t.Button(master,text="Camera",width=13, command=self.camera_command)
        self.browse_button = t.Button(master,text="Browse",width=13, command=self.browse_command)
        self.model_list = Listbox(master,height=10,width=50)
        self.label = Label(master,text="__Models__",font=("Lucida Console", 11))
        self.model_list_scorll = Scrollbar(master)
        self.crop_button_var = BooleanVar()
        self.crop_button_var.set(True)
        self.crop_button = Checkbutton(master,text="Image grid crop",
                state='active',onvalue = True, offvalue = False,
                variable=self.crop_button_var ,command=self.crop_button_command)
        self.config()
        self.align_components()
        self.model_list_show()
        self.selected_model = None
        self.camera_is_open = False
        self.classes = ['Without Mask','With Mask','Not a Person']
        self.transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def config(self):
        self.label.config(fg="#FFF")
        self.label.config(bg="#272727")
        self.model_list.config(fg="#FFF")
        self.model_list.config(bg="#353535")

        self.model_list.config(yscrollcommand=self.model_list_scorll.set)
        self.model_list.bind("<<ListboxSelect>>",self.get_item)
        self.model_list_scorll.config(command = self.model_list.yview)

    def model_list_show(self):
        models = os.listdir("./models/")
        for model in models:
            self.model_list.insert(END, model)

    def align_components(self):
        self.eval_button.grid(row=1,column=1,padx=5,pady=5)
        self.fm_eval_button.grid(row=1,column=2,padx=5,pady=5)
        self.female_eval_button.grid(row=1,column=3,padx=5,pady=5)
        self.male_eval_button.grid(row=1,column=4,padx=5,pady=5)
        self.camera_button.grid(row=1,column=5,padx=5,pady=5)
        self.browse_button.grid(row=1,column=6,padx=5,pady=5)
        self.crop_button.grid(row=2,column=3,sticky=W)
        self.model_list.grid(row=4,column=1,columnspan=5,padx=1,pady=1)
        self.label.grid(row=3,column=2)
        self.model_list_scorll.grid(row=4,column=6)

    def get_item(self,event):
        self.selected_model = self.model_list.get(self.model_list.curselection())

    def crop_button_command(self):
        print(f"here:{self.crop_button_var.get()}")
        toggle =  not self.crop_button_var.get()
        self.crop_button_var.set(value=toggle)
        self.crop_button.toggle()



    def eval_command(self, path):
        with torch.no_grad():
            dt = Dataset(path,(32,32))
            dt_loader = DataLoader(dt,batch_size=len(dt), shuffle=False)
            x_test,y_test = next(iter(dt_loader))
            if not torch.cuda.is_available():
                model = torch.load(f"./models/{self.selected_model}", map_location=torch.device('cpu'))
                model = model.cpu()
                model.eval()
                y_out = model(x_test)
                _, y_pred = torch.max(y_out.data, 1)
                evaluation = Torch_Evaluator(x_test,y_test,y_pred,model)
                evaluation.evaluate()
            else:
                model = torch.load(f"./models/{self.selected_model}")
                x_test,y_test = x_test.cuda(),y_test.cuda()
                model.eval()
                y_out = model(x_test).cuda()
                _, y_pred = torch.max(y_out.data, 1)
                evaluation = Torch_Evaluator(x_test,y_test,y_pred,model)
                evaluation.evaluate()

    def camera_command(self):
        if not self.camera_is_open:
            camera_window = Toplevel(self.master)

            width, height = 800, 600
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            def destroy():
                camera_window.quit()
                cv2.VideoCapture(1)
            camera_window.bind('<Escape>', lambda e: destroy)

            lmain = Label(camera_window)
            lmain.grid(row=1,column=1)
            txt = StringVar()
            l = Label(camera_window,textvariable=txt)
            l.grid(row=2,column=1)

            def show_frame():
                _, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                lmain.imgtk = imgtk
                lmain.configure(image=imgtk)

                batch = self.get_grid_batch(img,resize=128,crop_size= 16,step_size=16) if self.crop_button_var.get() else img
                label = self.eval_batch(batch,True)

                txt.set(label)
                lmain.after(10, show_frame)

            show_frame()


    def get_grid_batch(self,image,resize=512,crop_size=128,step_size=64):

        preds = []
        image = image.resize((resize,resize))
        w, h = image.size
        left = 0
        top = 0
        batch = []
        while (left < w):
            top = 0
            while (top < h):

                if (top + crop_size > h and left + crop_size > w):
                    bbox = (left, top, w, h)

                elif (left + crop_size > w):
                    bbox = (left, top, w, top + crop_size)

                elif (top + crop_size > h):
                    bbox = (left, top, left + crop_size, h)

                else:
                    bbox = (left, top, left + crop_size, top + crop_size)
                cimage = image.crop(bbox)
                batch.append(cimage)
                top+= step_size
            left+= step_size

        return batch

    def eval_batch(self, image_batch,print_count=False):
        preds = []
        if not torch.cuda.is_available():
            model = torch.load(f"./models/{self.selected_model}", map_location=torch.device('cpu'))
            device = torch.device('cpu')
            model = model.to(device=device)
        else:
            model = torch.load(f"./models/{self.selected_model}")
            device = torch.device('cuda')
            model = model.to(device=device)
        model.eval()
        if not isinstance(image_batch,list):
            cimage = image_batch
            cimage = self.transform(cimage).to(device=device)
            cimage = Variable(cimage)
            cimage = cimage.unsqueeze(0)
            out = model(cimage)
            _, prediction = torch.max(out.data,1)
            label = self.classes[prediction]
            return label

        for cimage in image_batch:
            cimage = self.transform(cimage).to(device=device)
            cimage = Variable(cimage)
            cimage = cimage.unsqueeze(0)
            out = model(cimage)
            _, prediction = torch.max(out.data,1)
            label = self.classes[prediction]
            preds.append(label)
        if print_count:
            print(f"wm count: {preds.count('With Mask')} \nnm count: {preds.count('Without Mask')}\nnp count: {preds.count('Not a Person')}")
        label = ""
        wm_count = preds.count("With Mask")
        nm_count = preds.count("Without Mask")
        np_count = preds.count('Not a Person')
        total = wm_count + nm_count + np_count
        wm_ratio,nm_ratio,np_ratio = wm_count/total, nm_count/total, np_count/total

        nm_score = 0.55 * nm_count
        wm_score = 0.35 * wm_count
        np_score = 0.015 * np_count
        if wm_score + nm_score + abs(0.2*(wm_count-nm_count)) >= np_score:
            label = "With Mask" if  wm_score >= nm_score else "Without Mask"
        else:
            label = "Not a Person"
        print(label)
        return label

    def browse_command(self):
        img_path = filedialog.askopenfilename(initialdir = "/", title = "Select Image")
        image = Image.open(img_path)
        preds = []
        image_batch = self.get_grid_batch(image) if self.crop_button_var.get() else image
        label = self.eval_batch(image_batch,print_count=True)
        result_window = Toplevel(self.master)
        lmain = Label(result_window)
        img = Image.open(img_path)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        result_label = Label(result_window,text = label)
        lmain.grid(row=1,column=1)
        result_label.grid(row=2,column=1)


if __name__=='__main__':
    root = Tk()
    root.resizable(0,0)
    root.title('NASA Classifier')
    app = GUI(root)
    root.mainloop()