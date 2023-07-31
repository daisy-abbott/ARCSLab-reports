from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pathlib
from os import path

from fastai.vision.all import *
from fastai.callback.progress import CSVLogger

import wandb

# Initialize W&B run (if not already initialized)
run = wandb.init(
    project="random_false_perfect_img-cmd-cur",
    entity="arcslaboratory",
    notes="without callback, testing on 2k images.",
    job_type="dataset-upload"
)

# Load the dataset artifact
artifact = run.use_artifact("arcslaboratory/wandering-random-2K+/07-26_wandering_10Trials_randomized:v0")
artifact_dir = artifact.download()

dataset_path = artifact_dir + '/'   # Path to the extracted images from the artifact

from torch.utils.data import Dataset

# Constants 
# Load the dataset artifact

NUM_REPLICATES = 2
NUM_EPOCHS = 1
# DATASET_DIR = Path("/data/clark/summer2021/datasets")
DATASET_DIR = Path(dataset_path)
MODEL_PATH_REL_TO_DATASET = Path("cmd_models_fixed")
DATA_PATH_REL_TO_DATASET = Path("cmd_data_fixed")
# VALID_MAZE_DIR = Path("../Mazes/validation_mazes8x8/")

compared_models = {
    "resnet18": resnet18,
    "resnet34": resnet34
}

def filename_to_class(filename: str) -> str:
    filename_str = str(filename)
    # For regular filenames with angle values
    angle = float(filename_str.split("/")[-1].split("_")[2].split(".")[0].replace("p", "."))
    # -5 and 5 is still forwrad, keep the window for going forward larger
    if angle > 0:
        return "left"
    elif angle < 0:
        return "right"
    else:
        return "forward"
    

class ImageWithCmdDataset(Dataset):
    def __init__(self, filenames, img_size):
        """
        Creates objects for class labels, class indices, and filenames.
        :param filenames: (list) a list of filenames that make up the dataset
        """
        self.class_labels = ['left', 'forward', 'right']
        self.class_indices = {lbl:i for i, lbl in enumerate(self.class_labels)} # {'left': 0, 'forward': 1, 'right': 2}
        self.all_filenames = filenames
        self.img_size = int(img_size)
        
    def __len__(self):
        """
        Gives length of dataset.
        
        :return: (int) the number of filenames in the dataset
        """
        return len(self.all_filenames)

    def __getitem__(self, index):
        """
        Gets the filename associated with the given index, opens the image at
        that index, then uses the image's filename to get information associated
        with the image such as its label and the label of the previous image.
        
        :param index: (int) number that represents the location of the desired data
        :return: (tuple) tuple of all the information associated with the desired data
        """
        # The filename of the image given a specific index
        img_filename = self.all_filenames[index]            
        
        # Opens image file and ensures dimension of channels included
        img = Image.open(img_filename).convert('RGB')
        # Resizes the image
        img = img.resize((self.img_size, self.img_size))
        # Converts the image to tensor and 
        img = torch.Tensor(np.array(img)/255)
        # changes the order of the dimensions
        img = img.permute(2,0,1)
        
        # Getting the current image's label
        label_name = filename_to_class(img_filename)
        label = self.class_indices[label_name]
        
        # Getting the previous image's label
        # The default is 'forward'
        cmd_name = 'forward'
        
        # If the index is not 0, the cmd is determined by the previous img_filename
        if index != 0:
            prev_img_filename = self.all_filenames[index-1]
            cmd_name = filename_to_class(prev_img_filename)            
        cmd = self.class_indices[cmd_name]
        
        # Data and the label associated with that data
        return (img, cmd), label

class cmd_model(nn.Module):
    def __init__(self, arch: str, pretrained: bool):
        super(cmd_model, self).__init__()
        self.cnn = arch(pretrained=pretrained)
        
        self.fc1 = nn.Linear(self.cnn.fc.out_features + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, data):
        # Unpack the data as image and command
        img, cmd = data
        # Pass the image data to the cnn
        x1 = self.cnn(img)
        # Returns a new tensor from the cmd data
        x2 = cmd.unsqueeze(1)
        # Concatenate cmd and image in the 1st dimension 
        x = torch.cat((x1, x2), dim=1)
        # Apply the ReLU function element-wise to the linearly transformed img+cmd data
        x = self.r1(self.fc1(x))
        # Apply the linear transformation to the data 
        x = self.fc2(x)
        return x

def get_fig_filename(prefix: str, label: str, ext: str, rep: int) -> str:
    fig_filename = f"{prefix}-{label}-{rep}.{ext}"
    print(label, "filename :", fig_filename)
    return fig_filename

def prepare_dataloaders(dataset_name: str, prefix: str, valid_pct: float, img_size: int) -> DataLoaders:
    
    path = DATASET_DIR / dataset_name
    files = get_image_files(path)
    
    # Get size of dataset and corresponding list of indices
    dataset_size = len(files)
    dataset_indices = list(range(dataset_size))
    
    # Shuffle the indices
    np.random.shuffle(dataset_indices)
    
    # Get the index for where we want to split the data
    val_split_index = int(np.floor(float(valid_pct) * dataset_size))
    
    # Split the list of indices into training and validation indices
    train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]
    
    # Get the list of filenames for the training and validation sets
    train_filenames = [files[i] for i in train_idx]
    val_filenames = [files[i] for i in val_idx]
    
    # Create training and validation datasets
    train_data = ImageWithCmdDataset(train_filenames, img_size)
#     train_data.__get_item__(10)
    val_data = ImageWithCmdDataset(val_filenames, img_size)
    
    # Get DataLoader
    dls = DataLoaders.from_dsets(train_data, val_data)
    dls = dls.cuda()
    
    #dls.show_batch()  # type: ignore
    plt.savefig(get_fig_filename(prefix, "batch", "pdf", 0))
    return dls  # type: ignore

def train_model(
    dls: DataLoaders,
    model_arch: str,
    pretrained: bool,
    logname: Path,
    modelname: Path,
    prefix: str,
    rep: int,
):
    arch = compared_models[model_arch]
    net = cmd_model(arch, pretrained=pretrained)
    
    learn = Learner(
        dls,
        net,
        loss_func=CrossEntropyLossFlat(),
        metrics=accuracy,
        cbs=CSVLogger(fname=logname),
    )

    
    if pretrained:
        learn.fine_tune(NUM_EPOCHS)
    else:
        learn.fit_one_cycle(NUM_EPOCHS)

  # Save trained model
    torch.save(net.state_dict(), modelname)  # Save only the model's state_dict
    
    # Remove callback function
    learn.remove_cb(CSVLogger)
    

def main():

    arg_parser = ArgumentParser("Train cmd classification networks.")
    arg_parser.add_argument(
        "model_arch", help="Model architecture (see code for options)"
    )
    arg_parser.add_argument(
        "dataset_name", help="Name of dataset to use (corrected-wander-full)"
    )
    arg_parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained model"
    )
    arg_parser.add_argument(
        "gpu", help="Assign GPU"
    )
    arg_parser.add_argument(
        "valid_pct", help="Validation percentage"
    )
    arg_parser.add_argument(
        "img_size", help="The size of image training data"
    )

    args = arg_parser.parse_args()

    # TODO: not using this (would require replacing first layer)
    # rgb_instead_of_gray = True

    # Assign GPU:
    torch.cuda.set_device(int(args.gpu))
    print("Running on GPU: " + str(torch.cuda.current_device()))

    # Make dirs as needed
    model_dir = DATASET_DIR / args.dataset_name / MODEL_PATH_REL_TO_DATASET 
    model_dir.mkdir(exist_ok=True)
    print(f"Created model dir (or it already exists) : '{model_dir}'")

    data_dir = DATASET_DIR / args.dataset_name / DATA_PATH_REL_TO_DATASET
    data_dir.mkdir(exist_ok=True)
    print(f"Created data dir (or it already exists)  : '{data_dir}'")

    file_prefix = "classification-" + args.model_arch
    # file_prefix += "-rgb" if rgb_instead_of_gray else "-gray"
    file_prefix += "-pretrained" if args.pretrained else "-notpretrained"
    fig_filename_prefix = data_dir / file_prefix

    dls = prepare_dataloaders(args.dataset_name, fig_filename_prefix, args.valid_pct, args.img_size)
    
    # Train NUM_REPLICATES separate instances of this model and dataset
    for rep in range(NUM_REPLICATES):
        model_filename = DATASET_DIR / args.dataset_name / MODEL_PATH_REL_TO_DATASET / f"{file_prefix}-{rep}.pkl"
        print("Model relative filename :", model_filename)
        
        # Checks if model exists and skip if it does (helps if this crashes)
        if path.exists(model_filename):
            continue
        log_filename = DATASET_DIR / args.dataset_name / DATA_PATH_REL_TO_DATASET / f"{file_prefix}-trainlog-{rep}.csv"
        print("Log relative filename   :", log_filename)
        
        train_model(
            dls,
            args.model_arch,
            args.pretrained,
            log_filename,
            model_filename,
            fig_filename_prefix,
            rep,
        )

        # Create the artifact to save the models
        artifact = wandb.Artifact(
        name="07-31-img-cmd-2k",
        type="model"
        )

        # Log the artifact to wandb
        artifact.add_dir(f"{model_dir}", name="models")
        run.log_artifact(artifact)

if __name__ == "__main__":
    main()