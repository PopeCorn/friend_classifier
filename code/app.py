import torch, torchvision.transforms as transforms
import customtkinter as ctk
from customtkinter import filedialog
import imghdr, sys
from model import Mojmyr
from PIL import Image

# CTK settings
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# Initialize the model
myr_model = Mojmyr(input_shape=3, hidden_units=30, output_shape=1)
myr_model.load_state_dict(torch.load('code/!model_0_state_dict.pth'))

# Define image transform that will be used on the input sample
transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

class App(ctk.CTk):
    def __init__(self, model):
        super().__init__()

        self.minsize(300, 200)
        self.model = model

        # Button for uploading the image
        self.button = ctk.CTkButton(self, 
                                    text="Upload your image\n(.jpeg, anything else will shut the app down)", 
                                    command=self.image_processing)
        self.button.grid(padx=20, pady=10, sticky="news")

    def image_processing(self):
        try:
            filename = filedialog.askopenfilename()
            file_type = imghdr.what(filename)

            if file_type == "jpeg":
                image = Image.open(filename)
                image_tensor = transform(image)
                results = self.predict(image_tensor)
                text = results[0][0].item() # Get to the actual boolean value in the tensor
                print(text)

            else:
                sys.exit() # The file type must be jpeg
        except FileNotFoundError:
            pass

    def predict(self, sample):
        with torch.inference_mode():
            logits = self.model(sample.unsqueeze(dim=0)) # sample.unsqueeze to get the right shape for the model
            pred = (torch.sigmoid(logits) > 0.5) # Convert logits to probabilites using sigmoid function, then to labels by setting a 0.5 treshold
            return pred

root = App(myr_model)
root.mainloop()