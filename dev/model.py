from torch import nn 

class Mojmyr(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        # Copy TinyVGG structure, modify it slightly for this specific case
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1), 
            nn.ReLU(), 
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*14*14, out_features=output_shape)
        )

    # Required forward method that takes the input 'x' through all the conv_blocks and the classifier, returning logits because of the last Linear layer
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x