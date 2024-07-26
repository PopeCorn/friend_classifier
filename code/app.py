import torch, torchvision.transforms as transforms
import customtkinter as ctk
from customtkinter import filedialog
import imghdr
from model import Mojmyr
from PIL import Image

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

myr_model = Mojmyr(input_shape=3, hidden_units=100, output_shape=1)
myr_model.load_state_dict(torch.load('!model_0_state_dict.pth'))

img_count = 0
guide = '''Short guide:
1. For good results, upload images of an actual face

2. The model is trained only on faces, it cannot properly
process anything else and will give non-sensical results. I wish
I could re-train it on more diverse samples, however, I no longer have
any of the original positive data.

3. Images in formats other than jpeg won't get uploaded.'''

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # in case of 1 colour channel 
    transforms.Resize(size=(64, 64)), # same size model was trained on
    transforms.ToTensor()])

class App(ctk.CTk):
    def __init__(self, model):
        super().__init__()

        self.minsize(400, 280)
        self.model = model

        self.guide = ctk.CTkLabel(self, text=guide, fg_color="transparent")
        self.guide.pack(padx=20, pady=10, side="top")

        self.button = ctk.CTkButton(self, text="Upload", command=self.upload)
        self.button.pack(padx=20, pady=10, side="top")

        self.label = ctk.CTkLabel(self, text=None, fg_color="transparent")

    def upload(self):
        global img_count
        try:
            filename = filedialog.askopenfilename()
            file_type = imghdr.what(filename)

            if file_type == "jpeg":
                img_count += 1
                image = Image.open(filename)
                image_tensor = transform(image)
                certainty = self.predict(image_tensor)

                stats = f'''Image count: {img_count}
Probability of MYR there: {certainty:.2f}%'''

                self.label.configure(text=stats)
                self.label.pack(padx=20, pady=10, side="top")

            self.update()

        except FileNotFoundError:
            pass

    def predict(self, sample):
        with torch.inference_mode():
            logits = self.model(sample.unsqueeze(dim=0)) # get right shape 
            prob_pred = torch.sigmoid(logits)[0][0].item()
            certainty = (prob_pred * 100)
            return certainty

root = App(myr_model)
root.mainloop()
