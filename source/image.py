from timeit import default_timer as timer
from shutil import rmtree
from pathlib import Path
import PIL
import os

from numpy.random import RandomState, SeedSequence
from numpy.random import MT19937
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2


def timer_decorator(func):
    def new_func(*args, **kwargs):
        start = timer()
        func(*args, **kwargs)
        return timer() - start
    return new_func


def get_path_size(func):
    def new_func(*args, **kwargs):
        result = func(*args, **kwargs)
        return result, os.path.getsize(args[1])

    return new_func


@get_path_size
@timer_decorator
def pil_timeit(array, path):
    return PIL.Image.fromarray(array).save(path, optimize=True)


@get_path_size
@timer_decorator
def pil_1bit_timeit(array, path):
    return PIL.Image.fromarray(array).save(path, bits=1, optimize=True)


@get_path_size
@timer_decorator
def cv2_timeit(array, path):
    return cv2.imwrite(path, array)


IMAGE_SIZES = (100, 1000, 2500, 3500)
COLOR_SPACES = ("greyscale", "1_bit", "rgb")
NUMBER_OF_IMAGES = 500

source_path = Path().resolve()
assert (benchmark_results_path := source_path.parent / "results" / "image").exists()
test_dir = source_path / "test_dir"
os.makedirs(test_dir)

rs1 = RandomState(MT19937(SeedSequence(123456789)))
times = {}
for resolution in IMAGE_SIZES:
    times[("rgb", resolution, "pil")] = []
    times[("rgb", resolution, "cv2")] = []
    times[("greyscale", resolution, "pil")] = []
    times[("greyscale", resolution, "cv2")] = []
    times[("1_bit", resolution, "pil")] = []
    times[("1_bit", resolution, "pil_1bit")] = []
    times[("1_bit", resolution, "cv2")] = []
    for i in range(NUMBER_OF_IMAGES):
        rgb_image = np.array(np.round(rs1.rand(resolution, resolution, 3) * 255), dtype=np.uint8)
        greyscale_image = np.array(np.round(rs1.rand(resolution, resolution) * 255), dtype=np.uint8)
        bit_image = np.array(np.round(rs1.rand(resolution, resolution)), dtype=bool)

        times[("rgb", resolution, "pil")].append(
            pil_timeit(rgb_image, str(test_dir / f"{i}_rgb_pil_{resolution}.png"))
        )
        times[("rgb", resolution, "cv2")].append(
            cv2_timeit(rgb_image, str(test_dir / f"{i}_rgb_cv2_{resolution}.png"))
        )

        times[("greyscale", resolution, "pil")].append(
            pil_timeit(greyscale_image, str(test_dir / f"{i}_greyscale_pil_{resolution}.png"))
        )
        times[("greyscale", resolution, "cv2")].append(
            cv2_timeit(greyscale_image, str(test_dir / f"{i}_greyscale_cv2_{resolution}.png"))
        )

        times[("1_bit", resolution, "pil")].append(
            pil_timeit(bit_image, str(test_dir / f"{i}_bit_pil_{resolution}.png"))
        )
        times[("1_bit", resolution, "pil_1bit")].append(
            pil_1bit_timeit(bit_image, str(test_dir / f"{i}_bit_pil_1bit_{resolution}.png"))
        )
        times[("1_bit", resolution, "cv2")].append(
            cv2_timeit(
                np.array(bit_image, dtype=np.uint8),
                str(test_dir / f"{i}_bit_cv2_{resolution}.png"),
            )
        )

rmtree(test_dir)

indexes = [i for i in range(1, NUMBER_OF_IMAGES + 1)]
new_record = []
df = pd.DataFrame(times, index=indexes)
df *= 1000
for index in df:
    new_record.append((*df[index].agg(["mean", "std"]).values, *index))

analysis_df = pd.DataFrame(
    new_record, columns=["mean", "std", "color_space", "resolution", "method"]
)

version_string = f"pil-{PIL.__version__}_cv2-{cv2.__version__}"
with pd.ExcelWriter(benchmark_results_path / f"images_{version_string}.ods") as writer:
    df.to_excel(writer, sheet_name="Data")
    analysis_df.to_excel(writer, sheet_name="Analysis")

grouped_by_color_space = analysis_df.groupby("color_space")

sns.set_theme(style="whitegrid")
fig = sns.catplot(
    data=analysis_df,
    col="color_space",
    kind="box",
    x="resolution",
    y="mean",
    hue="method",
    ci="std",
    palette="dark",
    alpha=0.6,
    height=6,
    sharey=False
)

fig.savefig(f"images_{version_string}.svg")
plt.show()
