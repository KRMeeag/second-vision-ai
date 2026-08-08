import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

print(fo.list_datasets())

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    dataset_name="sv-common-v1",
    classes=["Person", "Car", "Bus", "Waste Container"],
    label_types=["detections"],
    max_samples=200,
    shuffle=True,
    seed=42,
)

view = dataset.filter_labels(
    "ground_truth",
    F("IsDepiction") == False
)

session = fo.launch_app(view)
session.wait()