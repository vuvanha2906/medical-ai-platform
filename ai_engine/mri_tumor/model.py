from monai.networks.nets import SwinUNETR

def get_swin_unetr_model(device):
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=24,
        use_checkpoint=True,
        spatial_dims=3,
    ).to(device)

    return model
