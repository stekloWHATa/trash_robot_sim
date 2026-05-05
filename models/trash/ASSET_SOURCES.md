# Demo Trash Asset Sources

The current Gazebo demo uses downloaded GLB meshes from Poly Pizza. Poly Pizza
serves downloadable GLB/OBJ assets without login; this was used because
Sketchfab's official programmatic Download API requires OAuth/user
authentication.

## Downloaded Meshes

| Demo class | Local mesh | Source | Author | License |
|---|---|---|---|---|
| `aluminum_can` | `aluminum_can/meshes/soda_can_kenney.glb` | https://poly.pizza/m/KOLcdOATw6 | Kenney | Public Domain / CC0 1.0 |
| `plastic_bottle` | `plastic_bottle/meshes/water_bottle_quaternius.glb` | https://poly.pizza/m/KpxDpidn1Z | Quaternius | Public Domain / CC0 1.0 |
| `glass_bottle` | `glass_bottle/meshes/bottle_quaternius.glb` | https://poly.pizza/m/SU28ptta34 | Quaternius | Public Domain / CC0 1.0 |
| `cardboard_box` | `cardboard_box/meshes/cardboard_box_closed_kenney.glb` | https://poly.pizza/m/zv8NvYfT9B | Kenney | Public Domain / CC0 1.0 |
| `plastic_bag` | `plastic_bag/meshes/trash_bag_quaternius.glb` | https://poly.pizza/m/jYrMKg2Q7C | Quaternius | Public Domain / CC0 1.0 |
| `paper_packaging` | `paper_packaging/meshes/pizza_box_kenney.glb` | https://poly.pizza/m/PSqXX0pj5o | Kenney | Public Domain / CC0 1.0 |
| `cigarette_butt` | `cigarette_butt/meshes/cigarette_butt_poly_by_google.glb` | https://poly.pizza/m/bArNB49FTmO | Poly by Google | Creative Commons Attribution |

## Notes

- The old local OBJ/SDF primitive fallback files may remain in `meshes/`, but
  `model.sdf` now points at the downloaded `.glb` meshes.
- For CC-BY assets, keep the author/source line above in project documentation.
- If using Sketchfab/BlenderKit later, add the exact source URL, author,
  license and attribution requirements here before committing.
