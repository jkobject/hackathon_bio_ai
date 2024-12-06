import zarr
from tifffile import TiffFile
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Convert pyramidal TIFF into ZARR')
    parser.add_argument('--input', type=str, help='Path to source TIFF')
    parser.add_argument('--output', type=str, help='Path to dest ZARR')
    return parser.parse_args()

def main():
    args = parse_args()
    with TiffFile(args.input) as tif:
        # Create a Zarr group to store the pyramid
        zarr_group = zarr.open_group(args.output, mode='w')
        
        # Loop through each resolution level in the pyramid
        for level, page in enumerate(tif.pages):
            data = page.asarray()  # Read the data for this resolution
            resolution_group = zarr_group.create_dataset(
                str(level),  # Name each resolution
                data=data,             # Write the data
                chunks=(256, 256),     # Define chunk size (adjust as needed)
                compressor=zarr.Blosc(cname='zstd', clevel=5)  # Compression settings
            )

if __name__ == "__main__" :
    main()