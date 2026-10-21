Terrain textures: 1K diffuse maps from Poly Haven (https://polyhaven.com), CC0. Slugs match the file names
(cliff_side, brown_mud_dry, leafy_grass, coast_sand_01, snow_02, brick_wall_02, metal_plate, large_sandstone_blocks,
weathered_brown_planks, cobblestone_floor_08, dark_rock). Mapped triplanar in world space by src/client/terrainMaterial.js.
dark_rock is lightened from the download (`magick dark_rock.jpg -gamma 1.9 -modulate 104,110 dark_rock.jpg`): the
original averages 0.13 sRGB (0.015 linear), an albedo no tint can raise, and read as black in full sun.
