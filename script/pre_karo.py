# MIT License

# Copyright (c) 2024 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.




import os
from pathlib import Path
import sys 
import argparse
import json
import tqdm
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(".")
from thirdparty.colmap.pre_colmap import * 
from thirdparty.gaussian_splatting.utils.my_utils import rotmat2qvec
from thirdparty.gaussian_splatting.helper3dg import getcolmapsinglen3d
from thirdparty.gaussian_splatting.colmap_loader import read_extrinsics_binary, read_intrinsics_binary


def get_cam_name(video_path):
    return os.path.splitext(os.path.basename(video_path))[0] 


def prepare_colmap(folder, offset, extension, point_root, test_cams):
    test_cam_strings = ['/'+str(ID) for ID in test_cams] # turn into list of strings
    folderlist =  sorted(folder.iterdir())
    filtered = [entry for entry in folderlist if not np.any([str(entry).endswith(ID) for ID in test_cam_strings])] # only training images!

    savedir = point_root / f"colmap_{offset}" / "input"
    savedir.mkdir(exist_ok=True, parents=True)
        
    for folder in filtered :
        imagepath = folder / f"{str(offset).zfill(6)}.{extension}"
        imagesavepath = savedir / f"{folder.name}.{extension}"
        
        if imagesavepath.exists() or imagesavepath.is_symlink():
            continue
            
        imagesavepath.symlink_to(imagepath.resolve())
        # shutil.copy(imagepath, imagesavepath)

def create_cameras_file(md):
    cam_ids = md['cam_id'][0]
    cameras = []
    w2c = md['w2c'][0]
    for i,mat in enumerate(w2c):
        m = np.array(mat)
        colmapR = m[:3, :3] # rotation matrix
        T = m[:3, 3] # translation vector
        colmapQ = rotmat2qvec(colmapR) # rotation matrix to quaternion

        camera = {
            'id': cam_ids[i],
            'filename': f"{cam_ids[i]}.png",
            'w': md['w'],
            'h': md['h'],
            'fx': np.array(md['k'])[0,0,0,0],
            'fy': np.array(md['k'])[0,0,1,1],
            'cx': np.array(md['k'])[0,0,0,-1],
            'cy': np.array(md['k'])[0,0,1,-1],
            'q': colmapQ,
            't': T,
        }
        cameras.append(camera)
    return cameras

def write_colmap_single_cam(path, cameras, offset=0):
    projectfolder = path / f"colmap_{offset}"
    manualfolder = projectfolder / "manual"
    manualfolder.mkdir(exist_ok=True)

    savetxt = manualfolder / "images.txt"
    savecamera = manualfolder / "cameras.txt"
    savepoints = manualfolder / "points3D.txt"

    imagetxtlist = []
    cameratxtlist = []

    db_file = projectfolder / "input.db"
    if db_file.exists():
        db_file.unlink()

    db = COLMAPDatabase.connect(db_file)

    db.create_tables()


    for j,cam in enumerate(cameras):
        ID = cam['id']
        filename = cam['filename']

        # intrinsics
        w = cam['w']
        h = cam['h']
        fx = cam['fx']
        fy = cam['fy']
        cx = cam['cx']
        cy = cam['cy']

        # extrinsics
        colmapQ = cam['q']
        T = cam['t']

        # check that cx is almost w /2, idem for cy
        # assert abs(cx - w / 2) / cx < 0.10, f"cx is not close to w/2: {cx}, w: {w}"
        # assert abs(cy - h / 2) / cy < 0.10, f"cy is not close to h/2: {cy}, h: {h}"

        line = f"{ID} " + " ".join(map(str, colmapQ)) + " " + " ".join(map(str, T)) + f" {1} {filename}\n"
        imagetxtlist.append(line)
        imagetxtlist.append("\n")

        if j==0:
            params = np.array((fx , fy, cx, cy,))
            camera_id = db.add_camera(1, w, h, params)
            cameraline = f"{1} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n"
            cameratxtlist.append(cameraline)
        image_id = db.add_image(filename, camera_id,  prior_q=colmapQ, prior_t=T, image_id=ID)
        db.commit()
    db.close()

    savetxt.write_text("".join(imagetxtlist))
    savecamera.write_text("".join(cameratxtlist))
    savepoints.write_text("")  # Creating an empty points3D.txt file
    
    
def convert_selected_cam_matrix_to_colmapdb(path, offset=0,ref_frame=0,image_ext="png"):
    
    # 

    projectfolder = path / f"colmap_{offset}"
    refprojectfolder =  path / f"colmap_{ref_frame}"
    manualfolder = projectfolder / "manual"

    
    cameras_extrinsic_file = refprojectfolder / "distorted" / "sparse" / "0" / "images.bin" # from distorted?
    cameras_intrinsic_file = refprojectfolder / "distorted" / "sparse" / "0" / "cameras.bin"

    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    manualfolder.mkdir(exist_ok=True)     
    
    savetxt = manualfolder / "images.txt"
    savecamera = manualfolder / "cameras.txt"
    savepoints = manualfolder / "points3D.txt"
    imagetxtlist = []
    cameratxtlist = []

    db_file = projectfolder / "input.db"
    if db_file.exists():
        db_file.unlink()

    db = COLMAPDatabase.connect(db_file)

    db.create_tables()
        
    cam_extrinsics_by_name = {extr.name: extr for extr in cam_extrinsics.values()}
    
    videopaths = sorted((refprojectfolder / "images").glob(f"*.{image_ext}"))
    
    for i, videopath in enumerate(videopaths):
        filename = videopath.name #eg cam00.png
        extr = cam_extrinsics_by_name[filename]
        intr = cam_intrinsics[extr.camera_id]

        w, h, params = intr.width, intr.height, intr.params
        focolength = intr.params[0]
        fx, fy = intr.params[2], intr.params[3]

        colmapQ = extr.qvec
        T = extr.tvec
        

        ID = str(i+1)
        
        line = f"{ID} " + " ".join(map(str, colmapQ)) + " " + " ".join(map(str, T)) + f" {ID} {filename}\n"
        imagetxtlist.append(line)
        imagetxtlist.append("\n")

        camera_id = db.add_camera(4, w, h, params)
        cameraline = f"{ID} OPENCV {w} {h} {' '.join(params.astype(str))} \n"
        cameratxtlist.append(cameraline)
        
        image_id = db.add_image(filename, camera_id,  prior_q=colmapQ, prior_t=T, image_id=ID)
        db.commit()
    db.close()

    savetxt.write_text("".join(imagetxtlist))
    savecamera.write_text("".join(cameratxtlist))
    savepoints.write_text("")

if __name__ == "__main__" :
    """
    1. Move only training images for each frame to its own colmap project folder: "scene_root/point/colmap_*/"
    2. Retrieve the camera details from only one reference frame (they are static anyway), copy to all other frames in colmap DB format
    3. Run Colmap for each frame
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--imageext", default="png", type=str)
    parser.add_argument("--basedir", default="/workspace/input_data/rotation", type=str)
    parser.add_argument("--startframe", default=1, type=int) # Frames are numbered 1 to 48 
    parser.add_argument("--endframe", default=49, type=int)
    parser.add_argument("--refframe", default=1, type=int)
    args = parser.parse_args()

    # check image extension
    if args.imageext not in ["png","jpeg", "jpg"]:
        print("wrong extension")
        quit()
   
    # get input config
    image_ext = args.imageext
    base_dir = Path(args.basedir)
    start_frame_num = args.startframe
    end_frame_num = args.endframe
    duration = args.endframe - args.startframe
    pose_ref_frame_num = args.refframe

    # input checking
    if start_frame_num >= end_frame_num:
        print("start frame must smaller than end frame")
        quit()

    if not base_dir.exists():
        print("path not exist")
        quit()
    
    # path to images
    decoded_frame_root = base_dir / "ims"
    train_md = json.load((base_dir / "train_meta.json").open())
    test_md = json.load((base_dir / "test_meta.json").open())
    test_camera_ids = test_md['cam_id'][0]

    ## 1. prepare colmap input structure
    point_root = base_dir / "point"
    print("Start preparing colmap image input")
    for offset in range(start_frame_num, end_frame_num):
        prepare_colmap(decoded_frame_root, offset, image_ext, point_root, test_camera_ids)
        # Creates the desired folder structure for COLAMP by creating symlinks to the images
        # ONLY takes training images, not test images
    
    ## 2. Get the camera details and create colmap databases for each frame 
    cameras = create_cameras_file(train_md) # This is the same for each frame, since the cameras are all static
    colmap_project_root = base_dir / "point"
    for offset in tqdm.tqdm(range(start_frame_num, end_frame_num), desc="convert cam data to colmap db"):
        write_colmap_single_cam(colmap_project_root, cameras, offset)

    ## 3. Run Colmap for each frame, if error, reinstall opencv-headless 
    for offset in range(start_frame_num, end_frame_num):
        getcolmapsinglen3d(colmap_project_root, offset)