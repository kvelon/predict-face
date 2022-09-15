import numpy as np
import json
import pickle

def get_bboxes_dict(claims, token):
    """
    Args:
        claims (pd.DataFrame): Full dataframe of claims
        token (str): Token for the claim we want the bbox of

    Returns:
        dict: Dictionary where keys are frame numbers and 
                       values are maxem dicts
    """
    s = (claims['bbox'][claims['token'] == token]).values[0]
    s = s.replace("'", "\"")
    return json.loads(s)

def maxem_to_coords(maxem_dict):
    """ 
    Args:
        maxem_dict (dict): Dictionary with ['x_ctr', 'y_ctr', 'w', 'h'] keys
    
    Returns:
        boundaries (tuple): Tuple of (x_min, x_max, y_min, y_max)
    """
    return (
        maxem_dict["x_ctr"] - maxem_dict["w"] / 2,
        maxem_dict["x_ctr"] + maxem_dict["w"] / 2,
        maxem_dict["y_ctr"] - maxem_dict["h"] / 2,
        maxem_dict["y_ctr"] + maxem_dict["h"] / 2,
    )

def get_areas(bboxes):
    """
    Args:
        bboxes (dict): Dictionary where keys are frame numbers and 
                       values are maxem dicts
    Returns:
        areas (arr): Array of areas of bounding box of each frame
    """
    def maxem_to_area(maxem_dict):
        return int(maxem_dict['w']) * int(maxem_dict['h'])

    frames = len(bboxes.keys())
    areas = np.zeros(frames)
    for i, maxem_dict in enumerate(bboxes.values()):
        areas[i] = maxem_to_area(maxem_dict)
    return areas

def reorder_by_area(bboxes, increasing=True):
    """
    Args:
        bboxes (dict): Dictionary where keys are frame numbers and 
                       values are maxem dicts
    Returns:
        order (arr): Indices that would sort the areas array
    """
    areas = get_areas(bboxes)
    order = np.argsort(areas) if increasing else np.argsort(areas)[::-1]
    return order

# def get_reordered_frames(frames, order):
#     """
#     Args:
#         frames (arr): Sequence of frames
#         order (arr): Order to arrange the frames

#     Returns:
#         np.array: Frames rearranged according to order
#     """
#     return frames[order]

def videos_to_np(data_path, reorder_fn=None, save_filename=None):
    """
    Args:
        data_path (PosixPath): Path to directory which contains .pkl
        reorder_fn (function): Function that takes in bboxes and returns new order
        save_filename (str): Filename for saving output array
    Returns:
        np.array: num_videos x num_frames x height x width x channels
    """
    if reorder_fn is not None:
        file = open(data_path / "claims.pkl", 'rb')
        claims = pickle.load(file)
        file.close()

    file_names = list((data_path / 'videos').iterdir())
    shp = np.load(file_names[0])['data'].shape
    combined = np.zeros([len(file_names)] + list(shp), dtype=np.uint8)
    for i, f in enumerate(file_names):
        video = np.load(f)['data']
        if reorder_fn is not None:
            bboxes_dict = get_bboxes_dict(claims, f.stem)
            order = reorder_fn(bboxes_dict)
            combined[i] = video[order]
        else:
            combined[i] = video
    if save_filename is not None:
        np.save(data_path / save_filename, combined)

    return combined