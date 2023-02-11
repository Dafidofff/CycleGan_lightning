import yaml 

from pathlib import Path
from argparse import Namespace
from argparse import ArgumentParser

from pytorch_lightning import Trainer

from cycle_gan.models.discriminators import CycleGAN_Discriminator
from cycle_gan.models.unet_generators import CycleGAN_Unet_Generator
from cycle_gan.models.pl_cycle_gan import CycleGAN_LightningSystem

from cycle_gan.dataset.monet_dataset import MonetDataModule
from cycle_gan.augmentations.standard_transforms import BaselineImageTransform
from cycle_gan.utils.configs import load_config_from_yaml
from cycle_gan.utils.training import seed_everything, init_weights




def main(args: Namespace):
    # Set seed
    seed_everything(args.seed)

    # Get config file
    cfg = load_config_from_yaml(args.config_path)
    cfg.data.data_dir = Path(cfg.data.data_dir)

    # Initialize dataset and transforms
    transforms = BaselineImageTransform(img_size = cfg.data.im_size, prob = cfg.data.transforms.prob)
    dm = MonetDataModule(data_dir=cfg.data.data_dir, transform=transforms, batch_size=cfg.data.batch_size, split='train', seed=0)

    # Initialize models
    G_basestyle = CycleGAN_Unet_Generator()
    G_stylebase = CycleGAN_Unet_Generator()
    D_base = CycleGAN_Discriminator()
    D_style = CycleGAN_Discriminator()
    for net in [G_basestyle, G_stylebase, D_base, D_style]:
        init_weights(net, init_type='normal')

    # Initialize lightning system
    model = CycleGAN_LightningSystem(G_basestyle, G_stylebase, D_base, D_style, 
                                     cfg.training.lr_g, cfg.training.lr_d, transforms, 
                                     cfg.model.reconstr_w, cfg.model.id_w)


    # Initialize trainer
    trainer = Trainer(
        logger=False,
        max_epochs=cfg.training.max_epochs,
        accelerator="cpu",
        callbacks=None,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,  # Skip Sanity Check
    )

    # Train
    trainer.fit(model, datamodule=dm)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=Path, default="./configs/monet_default.yaml")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # 
    # parser = pl.Trainer.add_argparse_args(parser)


    args = parser.parse_args()
    main(args)