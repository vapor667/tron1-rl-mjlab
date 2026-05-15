from mjlab.terrains import FlatPatchSamplingCfg, TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxOpenStairsTerrainCfg,
    BoxRandomGridTerrainCfg,
    BoxRandomStairsTerrainCfg,
    BoxSteppingStonesTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    HfPerlinNoiseTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
)

SPAWN_PATCH_CFG = {
    "spawn": FlatPatchSamplingCfg(
        num_patches=32,
        patch_radius=0.45,
        max_height_diff=0.04,
        x_range=(-2.5, 2.5),
        y_range=(-2.5, 2.5),
    )
}

TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "flat": BoxFlatTerrainCfg(
            proportion=0.35,
        ),
        "boxes": BoxRandomGridTerrainCfg(
            proportion=0.06,
            grid_width=0.45,
            grid_height_range=(0.03, 0.12),
            platform_width=2.0,
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.10,
            flat_patch_sampling=SPAWN_PATCH_CFG,
            noise_range=(0.02, 0.08),
            noise_step=0.01,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.08,
            flat_patch_sampling=SPAWN_PATCH_CFG,
            slope_range=(0.0, 0.2),
            platform_width=2.0,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
        "hf_pyramid_slope_inverted": HfPyramidSlopedTerrainCfg(
            proportion=0.08,
            flat_patch_sampling=SPAWN_PATCH_CFG,
            slope_range=(0.0, 0.2),
            platform_width=2.0,
            inverted=True,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
        "wave_terrain": HfWaveTerrainCfg(
            proportion=0.06,
            flat_patch_sampling=SPAWN_PATCH_CFG,
            amplitude_range=(0.0, 0.1),
            num_waves=4,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
        "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.06,
            flat_patch_sampling=SPAWN_PATCH_CFG,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.3, 0.8),
            obstacle_height_range=(0.03, 0.12),
            num_obstacles=20,
            platform_width=2.0,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
            origin_z_offset=0.02,
        ),
        "perlin_noise": HfPerlinNoiseTerrainCfg(
            proportion=0.03,
            flat_patch_sampling=SPAWN_PATCH_CFG,
            height_range=(0.0, 0.1),
            octaves=4,
            persistence=0.3,
            lacunarity=2.0,
            scale=8.0,
            horizontal_scale=0.15,
            resolution=0.15,
            border_width=0.25,
        ),
        "open_stairs": BoxOpenStairsTerrainCfg(
            proportion=0.04,
            step_height_range=(0.05, 0.12),
            step_width_range=(0.3, 0.7),
            platform_width=1.5,
            border_width=0.25,
        ),
        "random_stairs": BoxRandomStairsTerrainCfg(
            proportion=0.04,
            step_width=0.8,
            step_height_range=(0.05, 0.12),
            platform_width=1.5,
            border_width=0.25,
        ),
    }
)

TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=8,
    curriculum=False,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "flat": BoxFlatTerrainCfg(
            proportion=0.125,
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.125,
            flat_patch_sampling=SPAWN_PATCH_CFG,
            noise_range=(0.02, 0.06),
            noise_step=0.01,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.125,
            flat_patch_sampling=SPAWN_PATCH_CFG,
            slope_range=(0.05, 0.25),
            platform_width=2.0,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
        "wave_terrain": HfWaveTerrainCfg(
            proportion=0.125,
            flat_patch_sampling=SPAWN_PATCH_CFG,
            amplitude_range=(0.04, 0.12),
            num_waves=4,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
        "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.125,
            flat_patch_sampling=SPAWN_PATCH_CFG,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.3, 0.8),
            obstacle_height_range=(0.04, 0.14),
            num_obstacles=18,
            platform_width=2.0,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
            origin_z_offset=0.02,
        ),
        "perlin_noise": HfPerlinNoiseTerrainCfg(
            proportion=0.125,
            flat_patch_sampling=SPAWN_PATCH_CFG,
            height_range=(0.02, 0.12),
            octaves=4,
            persistence=0.3,
            lacunarity=2.0,
            scale=8.0,
            horizontal_scale=0.15,
            resolution=0.15,
            border_width=0.25,
        ),
        "open_stairs": BoxOpenStairsTerrainCfg(
            proportion=0.125,
            step_height_range=(0.05, 0.12),
            step_width_range=(0.4, 0.7),
            platform_width=1.5,
            border_width=0.25,
        ),
        "random_stairs": BoxRandomStairsTerrainCfg(
            proportion=0.125,
            step_width=0.8,
            step_height_range=(0.05, 0.12),
            platform_width=1.5,
            border_width=0.25,
        ),
    },
)

TERRAINS_ENTITY_CFG = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TERRAINS_CFG,
    max_init_terrain_level=7,
    env_spacing=2.5,
)

TERRAINS_PLAY_ENTITY_CFG = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TERRAINS_PLAY_CFG,
    max_init_terrain_level=8,
    env_spacing=2.5,
)

PLANE_ENTITY_CFG = TerrainEntityCfg(
    terrain_type="plane"
)
