import numpy as np
from numba import njit

@njit(cache=True)
def fast_multi_lidar_scan(lidar_positions, ray_dirs, obs_centers, obs_radii, max_dist):
    """
    lidar_positions: [n_lidars, 2]
    ray_dirs: [n_rays_per_lidar, 2]
    obs_centers: [n_obs, 2]
    obs_radii: [n_obs]
    """
    n_lidars = lidar_positions.shape[0]
    n_rays = ray_dirs.shape[0]
    n_obs = obs_centers.shape[0]
    
    # results will be a flat array of shape [n_lidars * n_rays]
    results = np.ones(n_lidars * n_rays, dtype=np.float64)
    
    for l_idx in range(n_lidars):
        pos = lidar_positions[l_idx]
        
        for r_idx in range(n_rays):
            d = ray_dirs[r_idx]
            min_t = max_dist
            
            for o_idx in range(n_obs):
                center = obs_centers[o_idx]
                radius = obs_radii[o_idx]
                
                # vector from ray origin to obstacle center
                oc = pos - center
                
                # quadratic formula coefficients for ray-circle intersection
                # d - normalized direction vector of the ray
                b = np.dot(d, oc)
                c = np.dot(oc, oc) - radius**2
                
                discriminant = b*b - c
                if discriminant >= 0:
                    sqrt_d = np.sqrt(discriminant)
                    t = -b - sqrt_d
                    if 0.0 <= t < min_t:
                        min_t = t
                    else:
                        # check the other root
                        t = -b + sqrt_d
                        if 0.0 <= t < min_t:
                            min_t = t
            
            results[l_idx * n_rays + r_idx] = min_t / max_dist
            
    return results