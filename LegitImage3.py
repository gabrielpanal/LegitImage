import pickle
from tkinter import filedialog
import customtkinter
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt


from ELA import ErrorLevelAnalysis
from Maxpool2 import Maxpool
from Convolution2 import Conv
from keras.models import load_model
from matplotlib import pyplot as plt

customtkinter.set_appearance_mode("dark") 
customtkinter.set_default_color_theme("dark-blue")  

app = customtkinter.CTk()

app.title("LegitImage")

# Set up window
app.geometry("640x480")  
app.config(bg="#b0e5eb")
app.iconbitmap(r'Logo.ico')

class App():
    def __init__(self):

        self.imgPath = 'initial_image.png'

        # Frame Logo
        logoPath = Image.open('LogoT.png')
        bigLogo = customtkinter.CTkImage(dark_image=logoPath, size=(150, 210))
        self.label1 = customtkinter.CTkLabel(app, image=bigLogo, text="", bg_color="#b0e5eb")
        self.label1._image = bigLogo
        self.label1.pack(pady=60)

        # Get Started button
        self.button1 = customtkinter.CTkButton(app, text="Get Started", command=self.openMainFrame, corner_radius=5, bg_color="#b0e5eb")
        self.button1.pack(pady=10)
    
    def openMainFrame(self):
        # Destroy welcome frame
        self.label1.destroy()
        self.button1.destroy()

        # Left frame
        self.leftFrame = customtkinter.CTkFrame(app, width=256, height=512, bg_color="#b0e5eb", fg_color="#b0e5eb")
        self.leftFrame.pack_propagate(False)
        self.leftFrame.grid(row=0, column=0, padx=(60,0), pady=70, sticky="nsew")

        # Image frame
        self.imageFrame = customtkinter.CTkFrame(self.leftFrame)
        self.imageFrame.grid(row=0, column=0, pady=30, padx=5)

        # Load initial image
        pilImg = Image.open(self.imgPath)
        pilImg = pilImg.resize((256, 256))
        self.image = customtkinter.CTkImage(dark_image=pilImg, size=(256, 256))
        self.imageLabel = customtkinter.CTkLabel(self.leftFrame, image=self.image, text="")
        self.imageLabel.pack_propagate(False)
        self.imageLabel.pack(pady=(20,20))

        # Output label
        self.outputText = customtkinter.CTkLabel(self.leftFrame, text="", wraplength=256, text_color='black')
        self.outputText.pack_propagate(False)
        self.outputText.grid(row=1, column=0, pady=(10, 10))
        self.outputText2 = customtkinter.CTkLabel(self.leftFrame, text="", wraplength=256, text_color='black')
        self.outputText2.pack_propagate(False)
        self.outputText2.grid(row=2, column=0, pady=(10, 10))

        # Button frame
        self.rightFrame = customtkinter.CTkFrame(app, width=256, height=512, bg_color="#b0e5eb", fg_color="#b0e5eb")
        self.rightFrame.grid(row=0, column=1, padx=(80,0), pady=160, sticky="nsew")
        
        # Buttons
        insertImg = customtkinter.CTkButton(self.rightFrame, text="Insert Image", command=self.openFile)
        insertImg.pack(pady=(10, 10), padx=40)

        detectImg = customtkinter.CTkButton(self.rightFrame, text="Detect", command=self.detect)
        detectImg.pack(pady=(10, 106), padx=40)
    
    def openFile(self):
        # Open image
        self.imgPath = filedialog.askopenfilename()
        pilImg = Image.open(self.imgPath)
        pilImg = pilImg.resize((256, 256))
        self.image = ImageTk.PhotoImage(pilImg)
        self.image = customtkinter.CTkImage(dark_image=pilImg, size=(224, 224))
        self.imageLabel.configure(image=self.image)
        
    def detect(self):
        
        if(self.imgPath == 'initial_image.png'):
            messagebox.showwarning(title="Warning", message="No image inserted! Try again.")
        else:
            ela = ErrorLevelAnalysis()
            conv_model = load_model("conv_model_256.h5")

    ############################################## Preprocess ##############################################
            inputImg = Image.open(self.imgPath)
            grayscaled = inputImg.convert("L")
            
    ############################################## ELA ##############################################
            elaResult = ela.ELA(np.array(grayscaled), 70)
            resized = cv2.resize(elaResult, (256, 256))
            mean, std = ela.GetMeanStdPixels(resized)
            
    ############################################## Convolutional Filter ##############################################
            conv_kernel1 = conv_model.get_weights()[0]
            conv_kernel2 = conv_model.get_weights()[2]
            
            conv_kernel1 = np.array(conv_kernel1)
            conv_kernel2 = np.array(conv_kernel2)

            reshaped_kernel1 = np.transpose(conv_kernel1, (3, 2, 0, 1))  # Shape (8, 1, 3, 3)
            reshaped_kernel2 = np.transpose(conv_kernel2, (3, 2, 0, 1))  # Shape (16, 8, 3, 3)

            print(reshaped_kernel1.shape)  # (8, 1, 3, 3)
            print(reshaped_kernel2.shape)  # (16, 8, 3, 3)
            
            conv1 = Conv(numFilters=8, inputChannels=1, kernel=conv_kernel1)
            conv2 = Conv(numFilters=16, inputChannels=8, kernel=conv_kernel2)
            maxpool = Maxpool()
            
            output = conv1.Forward((resized/255) - 1)
            output = maxpool.Forward(output)
            output = conv2.Forward(output)
            output = maxpool.Forward(output)
            feature_map_max = np.max(output, axis=(0, 1))
            
            arr2_reshaped = np.reshape(np.array([mean, std]), (1, 2))
            arr1_reshaped = np.reshape(feature_map_max, (1, 16))
            
            print(arr2_reshaped.shape)
            print(arr1_reshaped.shape)
            
    ############################################## Feature Fusion ##############################################
            features = np.concatenate((arr2_reshaped, arr1_reshaped), axis=1)
            print(features.shape)
            print(features)
            
    ############################################## Naive Bayes ##############################################
            with open("model/nb10.pkl", 'rb') as file:
                naivebayes = pickle.load(file)

            prediction, prob = naivebayes.Predict(features)

            if(prediction == 1):
                self.outputText.configure(text="Manipulated!")
            else:
                self.outputText.configure(text="Authentic!")
                
            self.outputText2.configure(text=prob)

            self.resetBtn = customtkinter.CTkButton(self.rightFrame, text="Reset", fg_color='#A52A2A', hover_color='#7C3030', command=self.reset)
            self.resetBtn.pack(pady=(0, 30))

    def reset(self):
        # Reset the frame
        self.imgPath = 'initial_image.png'
        pilImg = Image.open(self.imgPath)
        pilImg = pilImg.resize((256, 256))
        self.image = customtkinter.CTkImage(dark_image=pilImg, size=(256, 256))
        self.imageLabel.configure(image=self.image)
        self.outputText.configure(text="")
        self.outputText2.configure(text="")
        
        self.resetBtn.destroy()
        

mainFrame = App()

app.mainloop()







