from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import random
import matplotlib.pyplot as plt


def read_traffic_hdf(filename):
    """Read known DCRNN traffic HDF layouts without relying on key auto-detect."""
    for key in (None, "df", "speed"):
        try:
            return pd.read_hdf(filename, key=key)
        except (KeyError, TypeError, ValueError):
            continue

    # Fallback for legacy fixed-format HDF files that pandas 3/PyTables may not
    # decode directly. DCRNN files store a single group named either `df` or
    # `speed` with axis0/axis1/block0_values arrays.
    import tables

    with tables.open_file(filename, mode="r") as h5:
        if hasattr(h5.root, "df"):
            group = h5.root.df
        elif hasattr(h5.root, "speed"):
            group = h5.root.speed
        else:
            raise ValueError(
                f"Unable to read traffic HDF file {filename}. "
                "Expected group 'df' or 'speed'."
            )

        columns = group.axis0.read()
        if getattr(columns, "dtype", None) is not None and columns.dtype.kind == "S":
            columns = [value.decode("utf-8") for value in columns]
        index = pd.to_datetime(group.axis1.read())
        values = group.block0_values.read()
        return pd.DataFrame(values, index=index, columns=columns)


def generate_graph_seq2seq_io_data(df, x_offsets, y_offsets
                                   , add_time_in_day=True, add_day_in_week=False):

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]

    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    random.seed(0)
    df = read_traffic_hdf(args.traffic_df_filename)


    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    y_offsets = np.sort(np.arange(1, args.pred_steps+1, 1))

    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    _x, _y = locals()["x"], locals()["y"]
    np.savez_compressed(
        os.path.join(args.output_dir, str(args.pred_steps) + "_series.npz"),
        x=_x,
        y=_y,
        x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
        y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_steps", type=int, default=1)
    parser.add_argument(
        "--output_dir", type=str, default="./PEMS-BAY", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename", type=str, default="pems-bay.h5", help="Raw traffic readings.",
    )
    args = parser.parse_args()
    print("Generating training data for {} with prediction steps={}".format(args.output_dir[2:], args.pred_steps))
    generate_train_val_test(args)
