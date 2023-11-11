import argparse
from trainer import Trainer

from model import Generator, Discriminator, GeneratorLarge, DiscriminatorLarge, AlternativeGenerator
import torchinfo


def main(args):
    if args.model == "normal":
        generator = Generator()
        discriminator = Discriminator()
    elif args.model == "large":
        generator = GeneratorLarge()
        discriminator = DiscriminatorLarge()
    elif args.model == "alt":
        generator = AlternativeGenerator()
        discriminator = Discriminator()
    else:
        pass

    torchinfo.summary(generator)
    torchinfo.summary(discriminator)

    trainer = Trainer(generator=generator, discriminator=discriminator, config=vars(args))
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAN.")

    parser.add_argument("--model", "-m", choices=["normal", "large", "alt"], default="normal", help="Choose the model")

    parser.add_argument(
        "--data-root",
        "-d",
        type=str,
        nargs="+",
        default=["dataset/crawled_cakes", "dataset/generated_cakes"],
        help="Path(s) to the root directory(ies) of the dataset. Multiple paths can be provided.",
    )

    parser.add_argument("--batch-size", "-bs", type=int, default=32, help="Batch size for training.")

    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate for training.")

    parser.add_argument("--l2", type=float, default=0.0001, help="Weight decay / l2 regularization.")

    parser.add_argument("--total-epochs", type=int, default=3000, help="Number of epochs for training.")

    parser.add_argument("--img-size", type=int, nargs=2, default=(32, 32), help="Img size.")

    parser.add_argument("--augment", action="store_true", help="Augment data.")

    parser.add_argument("--num-workers", type=int, default=8, help="num workers for dataloader.")

    parser.add_argument(
        "--snapshot-freq",
        type=int,
        default=100,
        help="Frequency (in terms of epochs) at which to save model snapshots.",
    )

    parser.add_argument(
        "--example-freq",
        type=int,
        default=20,
        help="Frequency (in terms of epochs) at which to save model snapshots.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to resume training from snapshot (if exists) or start from scratch.",
    )

    args = parser.parse_args()
    args.resume = not args.no_resume
    print(args)
    main(args)
