import argparse
from dataclasses import dataclass, asdict
from datam.dataloader import VSAVSmallDataset
from datam.litdataloader import VSAVDataModule

from torch.utils.data import DataLoader

def test_data(metadata_path, prefix, batch_size, index=0):
    train_dataloader = DataLoader(VSAVSmallDataset(metadata_path, prefix), batch_size, shuffle=True, collate_fn=VSAVDataModule.collate_fn_clip)

    batch = None
    for i, batch in enumerate(train_dataloader):
        if i > index:
            break
    
    return batch

def setup():
    parser = argparse.ArgumentParser(description="Setup dataset and model parameters")

    # Dataset / dataloader parameters
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to metadata file")
    parser.add_argument("--prefix", type=str, default="", help="Dataset prefix/folder")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for dataloader")
    parser.add_argument("--index", type=int, default=0, help="Which batch index to fetch")
    parser.add_argument("--verbose", type=bool, default=False, help="Print the values")

    return parser.parse_args()


if __name__ == "__main__":
    args= setup()
    batch = test_data(args.metadata_path, args.prefix, args.batch_size, args.index)
    print("Fetched batch:")
    if batch is not None:
        for item in batch:
            if args.verbose:
                print(item)
            print(item.shape)
        print(batch.shape)