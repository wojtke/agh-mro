import os
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CarrotDataset

import lovely_tensors as lt

lt.monkey_patch()

import wandb


class Trainer:
    def __init__(self, generator, discriminator, config):
        self.config = config
        self.epoch = 0
        self.batch_size = config["batch_size"]
        self.criterion = torch.nn.BCELoss()

        self.dataloader = DataLoader(
            CarrotDataset(config["data_root"], img_size=config["img_size"], augment=config["augment"]),
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=config["num_workers"],
            prefetch_factor=4,
        )

        self.example_noise = torch.normal(0, 1, [4, generator.fc.in_features]).cuda()

        self.generator = generator.cuda()
        self.discriminator = discriminator.cuda()

        self.gen_opt = optim.Adam(self.generator.parameters(), lr=config["lr"], weight_decay=config["l2"])
        self.dis_opt = optim.Adam(self.discriminator.parameters(), lr=config["lr"], weight_decay=config["l2"])

        run = wandb.init(project="gan-mro", config=config, resume=config["resume"])
        self.run_dir = os.path.join("runs", run.name)
        self.snapshot_path = os.path.join(self.run_dir, "training_snapshot.pt")
        os.makedirs(self.run_dir, exist_ok=True)

        if config["resume"] and os.path.exists(self.snapshot_path):
            print("Starting trainig from a snapshot")
            self.load_snapshot()
        else:
            print("Starting trainig from scratch")

    def save_snapshot(self):
        snapshot = {
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "gen_opt_state_dict": self.gen_opt.state_dict(),
            "dis_opt_state_dict": self.dis_opt.state_dict(),
            "epoch": self.epoch,
            "example_noise": self.example_noise,
        }
        torch.save(snapshot, self.snapshot_path)

    def load_snapshot(self):
        snapshot = torch.load(self.snapshot_path)

        self.generator.load_state_dict(snapshot["generator_state_dict"])
        self.discriminator.load_state_dict(snapshot["discriminator_state_dict"])
        self.gen_opt.load_state_dict(snapshot["gen_opt_state_dict"])
        self.dis_opt.load_state_dict(snapshot["dis_opt_state_dict"])
        self.epoch = snapshot["epoch"]
        self.example_noise = snapshot["example_noise"]

    @torch.no_grad()
    def save_examples(self):
        self.generator.eval()
        imgs = self.generator(self.example_noise)
        imgs = (imgs + 1) / 2
        imgs = [TF.to_pil_image(im) for im in imgs]

        wandb.log({"epoch": self.epoch, "examples": [wandb.Image(im) for im in imgs]})

    def save_generator(self):
        torch.save(self.generator.state_dict(), os.path.join(self.run_dir, f"gen_{self.epoch:05}.pt"))

    def train(self):
        pbar = tqdm(total=self.config["total_epochs"], initial=self.epoch, unit="epoch")

        while self.epoch < self.config["total_epochs"]:
            self.generator.train()
            self.epoch += 1

            running_loss_dis = 0
            running_loss_gen = 0
            for real_imgs in self.dataloader:
                real_imgs = real_imgs.cuda()
                real_labels = torch.normal(1, 0.05, (self.batch_size, 1)).cuda()
                # real_labels = torch.ones(self.batch_size, 1).cuda()
                noise = torch.normal(0, 1, [self.batch_size, self.generator.fc.in_features]).cuda()

                fake_imgs = self.generator(noise)
                fake_labels = torch.normal(0, 0.05, (self.batch_size, 1)).cuda()
                # fake_labels = torch.zeros(self.batch_size, 1).cuda()

                self.dis_opt.zero_grad()
                outputs_real = self.discriminator(real_imgs)
                loss_real = self.criterion(outputs_real, real_labels)

                outputs_fake = self.discriminator(fake_imgs.detach())
                loss_fake = self.criterion(outputs_fake, fake_labels)

                dis_loss = (loss_real + loss_fake) / 2

                dis_loss.backward()
                self.dis_opt.step()
                running_loss_dis += dis_loss.item()

                self.gen_opt.zero_grad()

                outputs_fake = self.discriminator(fake_imgs)
                gen_loss = self.criterion(outputs_fake, real_labels)
                gen_loss.backward()
                self.gen_opt.step()
                running_loss_gen += gen_loss.item()

                pbar.set_postfix(
                    gen=f"{gen_loss.item():.3f}",
                    dis=f"{dis_loss.item():.3f}",
                    refresh=True,
                )
            pbar.update()
            wandb.log(
                {
                    "epoch": self.epoch,
                    "dis_loss": running_loss_dis / len(self.dataloader),
                    "gen_loss": running_loss_gen / len(self.dataloader),
                }
            )

            if self.epoch % self.config["example_freq"] == 0:
                self.save_examples()

            if self.epoch % self.config["snapshot_freq"] == 0:
                self.save_snapshot()
                self.save_generator()

        wandb.finish()
