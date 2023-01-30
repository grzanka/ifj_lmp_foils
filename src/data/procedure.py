from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

from data.analysis import Circle, get_mean_std


def read_df(path : Path , det_names : List[str]) -> pd.DataFrame:
    df = pd.DataFrame()
    df["det_id"] = det_names
    df["raw_data"] = df.det_id.apply(lambda id: np.load(path / id / "raw.npy"))
    df["det_circle"] = df.det_id.apply(lambda x: Circle.from_json(path / f"{x}lv" / "det-circle.json"))
    df["raw_mean"] = df.apply(lambda tmpdf: get_mean_std(tmpdf.raw_data, tmpdf.det_circle)[0], axis=1)
    df["raw_std"] = df.apply(lambda tmpdf: get_mean_std(tmpdf.raw_data, tmpdf.det_circle)[1], axis=1)
    return df