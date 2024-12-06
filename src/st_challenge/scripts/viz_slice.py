import argparse
import napari
import zarr
import dask.array as da

def parse_args():
    parser = argparse.ArgumentParser(description='View a slice of a WSIs')
    parser.add_argument('name', type=str, help='Name of the WSI to view')
    return parser.parse_args()

def view_img(name):
    uri = f"s3://renier-bucket-dev2024/etienne.doumazane/wsis/{name}.zarr/"
    store = zarr.open(uri, mode="r")
    multiscale_data = [da.from_zarr(store[str(i)]) for i in range(7)]
    v = napari.Viewer(title=name)
    v.add_image(multiscale_data, name=name)
    v.scale_bar.visible = True
    v.scale_bar.unit = "px"
    return v

def main():
    args = parse_args()
    view_img(args.name)
    napari.run()


if __name__ == '__main__':
    main()
