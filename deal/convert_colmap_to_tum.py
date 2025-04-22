import numpy as np
from pyquaternion import Quaternion
import struct
from evo.core import lie_algebra, metrics
from evo.tools import file_interface, plot
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import imageio
# import skimage.transform
import collections
import numpy as np
import struct




CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    poses = []
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])# qw, qx, qy, qz
            tvec = np.array(binary_image_properties[5:8])# tx, ty, tz
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            # images[image_id] = Image(
            #     id=image_id, qvec=qvec, tvec=tvec,
            #     camera_id=camera_id, name=image_name,
            #     xys=xys, point3D_ids=point3D_ids)
            
            poses.append({
                "timestamp": image_id,
                "q_wc":  Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).normalised,
                "t_wc":tvec,
            })
    return poses



def load_colmap_data(realdir):
    
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm

def convert_colmap_to_tum(poses, output_path):
    """将 COLMAP 位姿（world-to-camera）转换为 TUM 格式（camera-to-world）"""
    with open(output_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for pose in poses:
            R_wc = pose["q_wc"].rotation_matrix
            t_wc = pose["t_wc"]
            T_wc = lie_algebra.se3(R_wc, t_wc)
            T_cw = lie_algebra.se3_inverse(T_wc)  # 求逆
            t_cw = T_cw[:3, 3]
            q_cw = Quaternion(matrix=T_cw[:3, :3]).normalised
            f.write(f"{pose['timestamp']} {t_cw[0]} {t_cw[1]} {t_cw[2]} "
                    f"{q_cw.x} {q_cw.y} {q_cw.z} {q_cw.w}\n")












if __name__ == "__main__":

    colmap_bin_path ="/home/slam/datasets/test/7_llff_test/new_colmap/images.bin"
    colmap_txt_path ="/home/slam/datasets/test/7_llff_test/new_colmap/colmap_tum_test.txt"

    # 读取 COLMAP 位姿并转换为 TUM 格式
    colmap_poses = read_images_binary(colmap_bin_path)
    convert_colmap_to_tum(colmap_poses,colmap_txt_path)
