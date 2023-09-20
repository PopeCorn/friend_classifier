import torch
import customtkinter as ctk
from customtkinter import filedialog
import imghdr
from model import Mojmyr

myr_model = Mojmyr(input_shape=3, hidden_units=10, output_shape=1)
input = None

# CTK and app settings
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.minsize(200,250)
        self.title = "MYR"

        ctk.CTkButton(self, text="Upload your image", command=self.UploadAction).grid(padx=20, pady=10, sticky="news")

    def UploadAction(self, event=None):
        filename = filedialog.askopenfilename()
        file_type = imghdr.what(filename)
        if file_type != "jpeg":
            pass
        else:
            pass

# REQUIRED SHAPE: torch.Size([1, 3, 64, 64])
def predict(model, sample):
    with torch.inference_mode():
        sample = torch.permute(sample, dims=(2, 0, 1))
        logits = model(sample.unsqueeze(dim=0))
        pred = (torch.sigmoid(logits) > 0.5)
    return pred

conversion = {0: "Mojmyr is not present",
              1: "Mojmyr is present"}

results = predict(myr_model, input)
text = conversion[results]