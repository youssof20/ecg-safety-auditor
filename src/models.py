import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        kernel_size2: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        padding2 = kernel_size2 // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size2, stride=1, padding=padding2, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet1D_12Lead(nn.Module):
    """
    12-lead ECG classifier (single-label superclass).

    Input:  (batch, 12, 1000)
    Output: (batch, 5)
    """

    def __init__(self, num_classes: int = 5, kernel_size: int = 7, dropout: float = 0.3):
        super().__init__()

        # 4 residual blocks with [64, 128, 128, 256] filters.
        # We downsample in later blocks to reduce temporal length and stabilize training.
        self.block1 = ResidualBlock1D(12, 64, kernel_size=kernel_size, stride=1, kernel_size2=kernel_size)
        self.block2 = ResidualBlock1D(64, 128, kernel_size=kernel_size, stride=2, kernel_size2=kernel_size)
        self.block3 = ResidualBlock1D(128, 128, kernel_size=kernel_size, stride=2, kernel_size2=kernel_size)
        self.block4 = ResidualBlock1D(128, 256, kernel_size=kernel_size, stride=2, kernel_size2=kernel_size)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

