import torch, torchvision.transforms as transforms
import customtkinter as ctk
from customtkinter import filedialog
import imghdr, sys
from model import Mojmyr
from PIL import Image


myr_model = Mojmyr(input_shape=3, hidden_units=30, output_shape=1)
myr_model.load_state_dict(torch.load('code/!model_0_state_dict.pth'))

transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

# CTK and app settings
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

def predict(model, sample):
    model.eval()
    with torch.inference_mode():
        logits = model(sample.unsqueeze(dim=0))
        pred = (torch.sigmoid(logits) > 0.5)
        return pred

class App(ctk.CTk):
    def __init__(self, model):
        super().__init__()

        self.minsize(300, 200)
        self.title = "MYR"
        self.text = None
        self.model = model

        ctk.CTkButton(self, text="Upload your image\n(.jpeg, anything else will shut the app down)", command=self.UploadAction).grid(padx=20, pady=10, sticky="news")

    def UploadAction(self):
        filename = filedialog.askopenfilename()
        file_type = imghdr.what(filename)

        if file_type == "jpeg":
            image = Image.open(filename)
            image_tensor = transform(image)
            results = predict(self.model, image_tensor)
            self.text = results[0][0].item()
            print(filename, self.text)
        else:
            sys.exit()

root = App(myr_model)
root.mainloop()