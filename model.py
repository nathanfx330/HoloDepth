import torch
import torch.nn as nn

class MobileConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(MobileConvBlock, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            return out + x
        else:
            return out

class HybridMPIModel(nn.Module):
    def __init__(self, num_views, num_planes):
        super(HybridMPIModel, self).__init__()
        c_in = 3 * num_views

        self.enc1 = MobileConvBlock(c_in, 32)
        self.enc2 = MobileConvBlock(32, 64, stride=2)
        self.enc3 = MobileConvBlock(64, 128, stride=2)
        self.enc4 = MobileConvBlock(128, 256, stride=2)

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(256, 128, kernel_size=1))
        self.dec1 = MobileConvBlock(256, 128)

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(128, 64, kernel_size=1))
        self.dec2 = MobileConvBlock(128, 64)
        
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(64, 32, kernel_size=1))
        self.dec3 = MobileConvBlock(64, 32)

        self.final_conv = nn.Conv2d(32, 4, kernel_size=1)
        self.num_planes = num_planes

    def forward(self, psv):
        b, v, d, h, w, c = psv.shape
        psv_fused = psv.permute(0, 2, 1, 5, 3, 4).reshape(b * d, v * c, h, w)
        
        e1 = self.enc1(psv_fused); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)

        d1 = torch.cat([self.up1(e4), e3], dim=1); d1 = self.dec1(d1)
        d2 = torch.cat([self.up2(d1), e2], dim=1); d2 = self.dec2(d2)
        d3 = torch.cat([self.up3(d2), e1], dim=1); d3 = self.dec3(d3)
        
        output_flat = self.final_conv(d3)
        
        mpi_rgb = output_flat[:, :3, :, :]
        mpi_alpha = torch.sigmoid(output_flat[:, 3:, :, :])
        
        mpi_rgba = torch.cat([mpi_rgb, mpi_alpha], dim=1)
        
        # ======================================================================
        # --- THE CONFIRMED FIX ---
        # Call .contiguous() before .view() to ensure correct memory layout for reshaping.
        # This prevents the bug where all depth planes become identical.
        mpi_final = mpi_rgba.contiguous().view(b, d, 4, h, w)
        # ======================================================================
        
        return mpi_final.permute(0, 1, 3, 4, 2)