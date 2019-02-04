from first_graphcut import GraphCut
from second_graphcut import second_graphcut, boundary_recognize
import warnings
from utils import *
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings("ignore")


def graphcut_main(root_path, output_root_path, weight_path):
    input_path = sorted((root_path / Path("input")).glob("*.tif"))
    detection_path = sorted((root_path / Path("detection")).glob("*.tif"))
    backprop_path = sorted((root_path / Path("backprop")).glob("*.tif"))
    phase_off = sorted((root_path / Path("phase_off")).glob("*.tif"))

    output_segmentation = output_root_path / Path("segmentation")
    output_instance = output_root_path / Path("instance")
    output_boundary = output_root_path / Path("boundary")
    output_color = output_root_path / Path("color")
    output_segmentation.mkdir(parents=True, exist_ok=True)
    output_instance.mkdir(parents=True, exist_ok=True)
    output_boundary.mkdir(parents=True, exist_ok=True)
    output_color.mkdir(parents=True, exist_ok=True)

    graph = GraphCut()

    for i, path in enumerate(zip(input_path, detection_path, backprop_path, phase_off)):
        graph.update_image(path)

        #  segmentation
        result = graph.graphcut()

        # instance_segmentation
        result = second_graphcut(path, result, weight_path=weight_path)
        # save_instance
        plt.imsave(
            str(output_color / Path("{:05}.tif".format(i))), result, format="tif"
        )
        cv2.imwrite(str(output_instance / Path("{:05d}.tif".format(i))), result)

        # save segmentation
        segmentation = np.zeros(result.shape)
        segmentation[result > 0] = 1
        cv2.imwrite(
            str(output_segmentation / Path("{:05d}.tif".format(i))),
            (segmentation * 255).astype(np.uint8),
        )

        # calculate boundary
        result = boundary_recognize(result)

        boundary = np.zeros(result.shape)
        boundary[result > 0] = 255
        cv2.imwrite(str(output_boundary / Path("{:05d}.tif".format(i))), boundary)
        print(i)


if __name__ == "__main__":
    plot_size = 12
    date = datetime.now().date()

    input_path = Path("../images/for_graphcut/test18_normalize")
    output_path = Path("./output/{}/test18".format(date))
    weight_path = Path("../weights/MSELoss/best_{}.pth".format(plot_size))

    graphcut_main(input_path, output_path, weight_path=weight_path)
