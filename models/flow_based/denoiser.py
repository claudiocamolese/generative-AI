import torch
import torch.nn as nn
import torch.nn.functional as F


class Denoiser(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, num_hiddens=[64, 128, 256]):
        super().__init__()

        # ---- DOWN (two downs) ----
        self.conv1 = nn.Conv2d(in_channels, num_hiddens[0], 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_hiddens[0])
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(num_hiddens[0], num_hiddens[0], 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_hiddens[0])
        self.act2 = nn.GELU()

        self.down1 = nn.Conv2d(num_hiddens[0], num_hiddens[1], 3, 2, 1)  # /2
        self.bn_down1 = nn.BatchNorm2d(num_hiddens[1])
        self.act_down1 = nn.GELU()

        self.down2 = nn.Conv2d(num_hiddens[1], num_hiddens[2], 3, 2, 1)  # /2
        self.bn_down2 = nn.BatchNorm2d(num_hiddens[2])
        self.act_down2 = nn.GELU()

        # ---- BOTTOM (adaptive) ----
        self.flatten = nn.AdaptiveAvgPool2d((1, 1))

        # expand back to original downsampled size dynamically,
        # so we don't hard-code 7×7 or 8×8.
        self.unflatten_conv = nn.Conv2d(num_hiddens[2], num_hiddens[2], 3, 1, 1)
        self.bn_unflat = nn.BatchNorm2d(num_hiddens[2])
        self.act_unflat = nn.GELU()

        # ---- CONDITIONING ----
        self.fc_class = nn.Sequential(
            nn.Linear(num_classes, num_hiddens[2]),
            nn.GELU(),
            nn.Linear(num_hiddens[2], num_hiddens[2]),
        )

        self.fc_time = nn.Sequential(
            nn.Linear(1, num_hiddens[2]),
            nn.GELU(),
            nn.Linear(num_hiddens[2], num_hiddens[2]),
        )

        # ---- UP ----
        self.up1 = nn.ConvTranspose2d(num_hiddens[2], num_hiddens[1], 4, 2, 1)  # ×2
        self.bn_up1 = nn.BatchNorm2d(num_hiddens[1])
        self.act_up1 = nn.GELU()

        self.up2 = nn.ConvTranspose2d(num_hiddens[1], num_hiddens[0], 4, 2, 1)  # ×2
        self.bn_up2 = nn.BatchNorm2d(num_hiddens[0])
        self.act_up2 = nn.GELU()

        # ---- FINAL ----
        self.out_conv = nn.Conv2d(num_hiddens[0], in_channels, 3, 1, 1)

    def forward(self, x, c, t, mask=None):
        # ---- DOWN ----
        d1 = self.act1(self.bn1(self.conv1(x)))
        d1 = self.act2(self.bn2(self.conv2(d1)))

        d2 = self.act_down1(self.bn_down1(self.down1(d1)))
        d3 = self.act_down2(self.bn_down2(self.down2(d2)))  # shape: B, C, H', W'

        # ---- BOTTOM ----
        b = self.flatten(d3)  # -> (B, C, 1, 1)

        # dynamic unflatten: match size of d3 (7×7 or 8×8)
        h, w = d3.shape[2], d3.shape[3]
        b = b.expand(-1, -1, h, w)
        b = self.act_unflat(self.bn_unflat(self.unflatten_conv(b)))

        # ---- CONDITIONING ----
        c_onehot = F.one_hot(c, num_classes=self.fc_class[0].in_features).float()
        if mask is not None:
            c_onehot = c_onehot * mask.unsqueeze(1)

        cond_c = self.fc_class(c_onehot).view(-1, b.size(1), 1, 1)
        cond_t = self.fc_time(t.unsqueeze(1)).view(-1, b.size(1), 1, 1)
        b = b + cond_c + cond_t

        # ---- UP ----
        u1 = self.act_up1(self.bn_up1(self.up1(b)))
        u2 = self.act_up2(self.bn_up2(self.up2(u1)))

        # ---- OUTPUT ----
        return self.out_conv(u2)
