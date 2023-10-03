
import wandb

wandb.login()

def upload_dataset_to_wandb(local_path, wandb_project_name, artifact_name):
    # Log into W&B
    # wandb.login()

    # Initialize wandb run specifiyinig names and destination location
    run = wandb.init(
        project=wandb_project_name,
        entity="arcslaboratory",  
        job_type="upload-dataset"
    )

    # Createartifact object
    artifact = wandb.Artifact(name=artifact_name, type="dataset")

    # Add file to the artifact's contents
    artifact.add_dir(local_path)

    # Save the artifact 
    run.log_artifact(artifact)

    # Finish the run
    run.finish()

# Usage
local_path = "data/WanderingStaticTextures"  
wandb_project_name = "DoWellDatasets"
artifact_name = "Wandering-Static-Textures"

upload_dataset_to_wandb(local_path, wandb_project_name, artifact_name)