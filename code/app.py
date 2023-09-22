import torch, torchvision.transforms as transforms
import customtkinter as ctk
from customtkinter import filedialog
import imghdr, sys
from model import Mojmyr
from PIL import Image

# CTK settings
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# Initialize the model, set global variables to be used throughout the code
myr_model = Mojmyr(input_shape=3, hidden_units=100, output_shape=1)
myr_model.load_state_dict(torch.load('code/!model_0_state_dict.pth'))

conversion = {True: "MYR IS THERE 🗿", False: "MYR IS NOT THERE"}
image_count = 0
g = '''Short guide: 
For accurate predictions, the image 
should be of a human face with approximately 
same sides. If the prediction is wrong, it is
probably because one of these requirements was
not fulfilled.

But the case can also be that the model just cannot
predict the image you provided correctly, as its testing
accuracy cannot get over about 97.5% - be aware of that
when uploading an image that is not Mojmyr but looks very 
much like it is.'''

# Define image transform that will be used on the input sample
transform = transforms.Compose([transforms.Resize(size=(64, 64)), transforms.ToTensor()])

class App(ctk.CTk):
    def __init__(self, model):
        super().__init__()

        self.minsize(500, 350)
        self.model = model

        self.guide = ctk.CTkLabel(self, text=g, fg_color="transparent").pack(padx=20, pady=10, side="top")

        # Button for uploading the image
        self.button = ctk.CTkButton(self, text="Upload your image (jpeg)", command=self.image_processing)
        self.button.pack(padx=20, pady=10, side="top")

        # Label for model's prediction result
        self.label = ctk.CTkLabel(self, text=None, fg_color="transparent")

    def image_processing(self):
        try:
            filename = filedialog.askopenfilename()
            file_type = imghdr.what(filename)

            if file_type == "jpeg":
                global image_count, d
                image_count += 1
                image = Image.open(filename)
                image_tensor = transform(image)
                results, certainty = self.predict(image_tensor)

                if self.label != None: # Update the prediction result
                    self.label.configure(text=f"Image: {image_count}\n{conversion[results]}\nCertainty: {certainty:.2f}%")
                self.label.pack(padx=20, pady=10, side="top")

            else:
                pass
            
            # Update the window
            self.update()
            self.update_idletasks()

        except FileNotFoundError:
            pass

    def predict(self, sample):
        with torch.inference_mode():
            logits = self.model(sample.unsqueeze(dim=0)) # sample.unsqueeze to get the right shape for the model
            prob_pred = torch.sigmoid(logits)[0][0].item()
            pred = (prob_pred > 0.5) # Convert logits to probabilites using sigmoid function, then to labels by setting a 0.5 treshold
            certainty = (prob_pred * 100) if pred == 1 else (100 - prob_pred * 100)
            return pred, certainty

root = App(myr_model)
root.mainloop()