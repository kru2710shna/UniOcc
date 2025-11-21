import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = "datasets/NuScenes-via-Occ3D-2Hz-mini/scene-0061/0.npz"
data = np.load(path, allow_pickle=True)

print("Keys:", data.files)

# ============================
# 1) PRINT METADATA
# ============================
for k in data.files:
    arr = data[k]
    print(f"{k}: type={type(arr)}, dtype={arr.dtype if hasattr(arr, 'dtype') else 'N/A'}")

    if isinstance(arr, np.ndarray):
        print("   shape:", arr.shape)

    if hasattr(arr, "dtype") and arr.dtype == object:
        print("   This key contains Python objects.")
        print("   Number of elements:", len(arr))
        print("   Example element:", arr[0])


# =======================================================
# 2) VISUALIZE 2D SLICES OF VOXEL GRID (top-down slices)
# =======================================================

occ = data["occ_label"]   # (200,200,16)

# pick a height slice
h = occ.shape[2] // 2     # mid-height slice
slice_top = occ[:, :, h]

plt.figure(figsize=(6,6))
plt.title("Occupancy Grid (Top-Down Slice @ mid height)")
plt.imshow(slice_top, cmap="nipy_spectral")
plt.colorbar(label="Semantic Class ID")
plt.show()


# =======================================================
# 3) VISUALIZE LIDAR MASK
# =======================================================

mask_lidar = data["occ_mask_lidar"][:, :, h]

plt.figure(figsize=(6,6))
plt.title("LiDAR Mask (Top-Down @ mid height)")
plt.imshow(mask_lidar, cmap="gray")
plt.colorbar(label="Visibility (1 = visible)")
plt.show()


# =======================================================
# 4) VISUALIZE CAMERA MASK
# =======================================================

mask_cam = data["occ_mask_camera"][:, :, h]

plt.figure(figsize=(6,6))
plt.title("Camera Mask (Top-Down @ mid height)")
plt.imshow(mask_cam, cmap="gray")
plt.colorbar(label="Visibility (1 = visible)")
plt.show()


# =======================================================
# 5) SIMPLE 3D VISUALIZATION OF OCCUPIED VOXELS
# =======================================================
# We will plot all occupied voxels (semantic != 10)

occ_grid = data["occ_label"]
occupied = occ_grid != 10   # binary occupancy

xs, ys, zs = occupied.nonzero()    # coordinates of occupied voxels

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, s=1, c='red')

ax.set_title("3D Occupancy Voxels (red = occupied)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()


# =======================================================
# 6) VISUALIZE FORWARD FLOW (2D slice)
# =======================================================

flow_fwd = data["occ_flow_forward"][:, :, h, :]  # take height slice

flow_magnitude = np.linalg.norm(flow_fwd, axis=-1)

plt.figure(figsize=(6,6))
plt.title("Flow Magnitude (Top-Down Slice)")
plt.imshow(flow_magnitude, cmap="viridis")
plt.colorbar(label="Flow speed")
plt.show()
