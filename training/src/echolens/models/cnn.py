import torch.nn as nn


class AudioCNN(nn.Module):
    """Compact CNN for log-mel spectrogram classification.

    Kept small on purpose: target is <50ms inference in the browser.
    AdaptiveAvgPool2d(1) lets the model accept any number of time frames T,
    which simplifies real-time streaming inference later on.
    """

    def __init__(self, n_classes: int = 10):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):  # x: (B, 1, 64, T)
        return self.head(self.features(x))  # logits (B, n_classes)
