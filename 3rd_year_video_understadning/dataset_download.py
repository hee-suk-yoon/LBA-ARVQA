from huggingface_hub import hf_hub_download, snapshot_download
import pandas as pd
import huggingface_hub
import ipdb

# download the parquet file

REPO_ID = "tomg-group-umd/cinepile"
DIR_PATH = "/data/kakao/workspace/dataset/cinepile"

# dataset = pd.read_csv(
# huggingface_hub.snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=DIR_PATH) 
# )

# =============================================================================================================
# transform the parquet file to a pandas dataframe
DATA_PATH = "/data/kakao/workspace/dataset/cinepile/data/train-00000-of-00003.parquet"
dataset = pd.read_parquet(DATA_PATH)

ipdb.set_trace()