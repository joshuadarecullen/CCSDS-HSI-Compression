import torch
import sys
from pathlib import Path
import rasterio

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.ccsds import SpectralPredictor

def raster_scan(image: torch.Tensor, predictor: SpectralPredictor):

    max_errors = None

    Z, Y, X = image.shape
    predictions = torch.zeros_like(image)
    residuals = torch.zeros_like(image)

    # Initialize sample representatives - these are computed during compression
    sample_representatives = torch.zeros_like(image)

    # Default to lossless compression if no max_errors provided
    if max_errors is None:
        max_errors = torch.zeros_like(image)

    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                # For the first sample, initialize sample representative with original value
                if z == 0 and y == 0 and x == 0:
                    sample_representatives[z, y, x] = image[z, y, x]

                pred = predictor.predict_sample(image,
                                                sample_representatives,
                                                z, y, x)
                predictions[z, y, x] = pred

                # Compute prediction residual
                residual = image[z, y, x] - pred
                residuals[z, y, x] = residual
                # Quantize the residual (simplified - assuming lossless for now)
                max_error = max_errors[z, y, x].int().item() if max_errors is not None else 0
                if max_error == 0:
                    quantizer_index = residual  # Lossless: q_z(t) = \Delta_z(t)
                else:
                    # Near-lossless quantization (simplified)
                    step_size = 2 * max_error + 1
                    quantizer_index = torch.round(residual / step_size)

                # Compute sample representative for this sample
                sample_rep = predictor._compute_sample_representative(
                    image[z, y, x], pred, quantizer_index, max_error
                )
                print(f'sample_rep: {sample_rep}')
                sample_representatives[z, y, x] = sample_rep

    return predictions


if __name__ == "__main__":

    num_bands = 8
    dynamic_range = 16
    prediction_bands = 2
    prediction_mode = 'full'
    local_sum_type = 'neighbor_oriented'
    use_narrow_local_sums = False

    predictor = SpectralPredictor(num_bands=num_bands,
                                  dynamic_range=dynamic_range,
                                  prediction_bands=prediction_bands,
                                  prediction_mode=prediction_mode,
                                  local_sum_type=local_sum_type,
                                  use_narrow_local_sums=use_narrow_local_sums,
                                  offset=2**(2) - 1)


if __name__ == "__main__":

    # Replace 'path_to_your_tif_file.tif' with the actual path to your TIFF file
    tif_path = '/home/joshua/Documents/phd_university/code/avaris_data/aviris_hyperspectral_data/19920612_AVIRIS_IndianPine_Site3.tif'

    # Open the TIFF file
    with rasterio.open(tif_path) as src:
        # Read the entire file into a numpy array
        cube = src.read()

    preds = raster_scan(torch.from_numpy(cube[:4,:8,:8]).float(), predictor)

    print(cube[0, 0, 0])
    print(preds[0, 0, 0].item() == cube[0, 0, 0].item())


    #TODO:
    # calculate t, the character count of the rater scan. instead of (row 3, column 4), it would be t=12
