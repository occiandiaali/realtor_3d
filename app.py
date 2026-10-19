# import os
# import io
# import shutil
# import numpy as np
# import cv2
# import open3d as o3d
# from fastapi import FastAPI, Request, File, UploadFile
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from fastapi.responses import HTMLResponse
# from typing import List

# app = FastAPI()

# # Mount static and template directories
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# UPLOAD_DIR = "static/uploads"
# OUTPUT_DIR = "static/models"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# @app.get("/", response_class=HTMLResponse)
# async def serve_landing(request: Request):
#     return templates.TemplateResponse(request=request, name="index.html")

# @app.post("/process-room", response_class=HTMLResponse)
# async def process_images(request: Request, photos: List[UploadFile] = File(...)):
#     # 1. Clear previous session & save uploaded images
#     shutil.rmtree(UPLOAD_DIR)
#     os.makedirs(UPLOAD_DIR, exist_ok=True)
    
#     image_paths = []
#     for index, photo in enumerate(photos):
#         file_path = os.path.join(UPLOAD_DIR, f"frame_{index:03d}.jpg")
#         with open(file_path, "wb") as f:
#             f.write(await photo.read())
#         image_paths.append(file_path)

#     # 2. Open3D Point-Cloud Reconstruction Engine
#     pcd = o3d.geometry.PointCloud()
    
#     # Process features across overlapping camera pairs
#     for i in range(len(image_paths)):
#         img = cv2.imread(image_paths[i])
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         h, w, _ = img.shape
        
#         # Synthetic intrinsic estimation (assuming standard FOV)
#         fx = fy = max(h, w)
#         cx, cy = w / 2, h / 2
#         intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        
#         # Estimate coarse structural depth from spatial intensity
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         depth_sim = cv2.Laplacian(gray, cv2.CV_64F)
#         depth_sim = cv2.normalize(depth_sim, None, 0.5, 3.0, cv2.NORM_MINMAX)
        
#         o3d_color = o3d.geometry.Image(img_rgb)
#         o3d_depth = o3d.geometry.Image(depth_sim.astype(np.float32))
        
#         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#             o3d_color, o3d_depth, convert_rgb_to_intensity=False
#         )
        
#         temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
#         # Rotate point clouds sequentially to represent circular room capture
#         angle = (2 * np.pi / len(image_paths)) * i
#         R = temp_pcd.get_rotation_matrix_from_xyz((0, angle, 0))
#         temp_pcd.rotate(R, center=(0, 0, 0))
#         temp_pcd.translate((np.sin(angle) * 1.5, 0, np.cos(angle) * 1.5))
        
#         pcd += temp_pcd

#     # Clean outlier noise
#     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#     pcd = pcd.select_by_index(ind)
    
#     # Export model to standard OBJ container
#     output_mesh_path = os.path.join(OUTPUT_DIR, "room_environment.ply")
#     o3d.io.write_point_cloud(output_mesh_path, pcd)

#     # 3. Return HTMX Fragment containing the 3D Viewer Container
#     return templates.TemplateResponse(
#         request=request,
#         name="viewer_fragment.html", 
#         context={"model_url": "/static/models/room_environment.ply"}
#     )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
import os
import shutil
from typing import List

import cv2
import numpy as np
import open3d as o3d
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "static/uploads"
OUTPUT_DIR = "static/models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Placeholder data for past 3D room scans
PAST_SCANS = [
    {
        "id": "scan_01",
        "title": "Modern Living Room",
        "date": "2026-10-15",
        "frames": 12,
        "thumbnail": "https://images.unsplash.com/photo-1600210492486-724fe5c67fb0?auto=format&fit=crop&w=400&q=80",
    },
    {
        "id": "scan_02",
        "title": "Minimalist Office",
        "date": "2026-10-18",
        "frames": 16,
        "thumbnail": "https://images.unsplash.com/photo-1524758631624-e2822e304c36?auto=format&fit=crop&w=400&q=80",
    },
    {
        "id": "scan_03",
        "title": "Primary Bedroom Enclosure",
        "date": "2026-10-19",
        "frames": 14,
        "thumbnail": "https://images.unsplash.com/photo-1616594039964-ae9021a400a0?auto=format&fit=crop&w=400&q=80",
    },
]


@app.get("/", response_class=HTMLResponse)
async def serve_landing(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"past_scans": PAST_SCANS}
    )


@app.post("/process-room", response_class=HTMLResponse)
async def process_images(request: Request, photos: List[UploadFile] = File(...)):
    # 1. Clear previous uploads & save current batch
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    image_paths = []
    for index, photo in enumerate(photos):
        file_path = os.path.join(UPLOAD_DIR, f"frame_{index:03d}.jpg")
        with open(file_path, "wb") as f:
            f.write(await photo.read())
        image_paths.append(file_path)

    if not image_paths:
        return HTMLResponse("<p style='color:red;'>No images received.</p>")

    # 2. Point-Cloud Reconstruction Engine using Open3D
    pcd = o3d.geometry.PointCloud()

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # Estimate intrinsics based on focal length estimation
        fx = fy = max(h, w)
        cx, cy = w / 2, h / 2
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        # Depth estimation simulation via spatial feature gradient
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        depth_sim = cv2.Laplacian(gray, cv2.CV_64F)
        depth_sim = cv2.normalize(depth_sim, None, 0.5, 3.0, cv2.NORM_MINMAX)

        o3d_color = o3d.geometry.Image(img_rgb)
        o3d_depth = o3d.geometry.Image(depth_sim.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, convert_rgb_to_intensity=False
        )

        temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsic
        )

        # Apply positional rotations for circular camera placement
        angle = (2 * np.pi / len(image_paths)) * i
        R = temp_pcd.get_rotation_matrix_from_xyz((0, angle, 0))
        temp_pcd.rotate(R, center=(0, 0, 0))
        temp_pcd.translate((np.sin(angle) * 1.5, 0, np.cos(angle) * 1.5))

        pcd += temp_pcd

    # Filter statistical outliers
    if len(pcd.points) > 0:
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        pcd = pcd.select_by_index(ind)

    # Save reconstructed point cloud
    output_mesh_path = os.path.join(OUTPUT_DIR, "room_environment.ply")
    o3d.io.write_point_cloud(output_mesh_path, pcd)

    return templates.TemplateResponse(
        request=request,
        name="viewer_fragment.html",
        context={"model_url": "/static/models/room_environment.ply"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)