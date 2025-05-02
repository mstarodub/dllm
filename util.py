import os
import torch
import wandb


def device():
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dl_workers():
  return 12 if torch.cuda.is_available() else 0


def grad_debug():
  torch.autograd.set_detect_anomaly(True)


def settings():
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def log(log_dict):
  if wandb.run is not None:
    wandb.log(log_dict)
  else:
    print(*log_dict.items(), sep='\n')


def save(checkpoint, step, path):
  torch.save(checkpoint, path)
  if wandb.run is not None:
    artefact = wandb.Artifact(f'checkpoint_step_{step}', type='model')
    artefact.add_file(path)
    wandb.run.log_artifact(artefact)


def wandb_download_checkpoint(run_id, step, checkpoint_dir):
  api = wandb.Api()
  run = api.run(f'mxst-university-of-oxford/lldm/{run_id}')
  artifacts = run.logged_artifacts()
  checkpoint_artifact = next(
    (a for a in artifacts if a.name == f'checkpoint_step_{step}:v0'), None
  )
  if checkpoint_artifact is not None:
    checkpoint_artifact.download(root=checkpoint_dir)
    print(f'downloaded checkpoint for step {step} from run {run_id}')
  else:
    print(f'no checkpoint for step {step} found in run {run_id}')


class Config:
  def __init__(self, d):
    self.cf_dict = d
    if d is not None:
      for key, value in d.items():
        setattr(self, key, value)
