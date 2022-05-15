import os
from time import strftime

# preprocessed_df_path = os.path.join(
#     "output",
# )
root_names = ["331020_afp_newhits"]  # "336505_afp_newhits": None}
df_names = root_names
memory_chunk_size = "100 MB"
final_results_path = os.path.join(
    f"results_{strftime('%Y%m%d-%H%M%S')}",
)
path_to_root_dict = os.path.join("input_root")
preprocess_branches = ["evN", "hits", "hits_row", "hits_col", "hits_q"]
preprocess_functions = [
    "add_hits_number",
    "add_average_coordinates",
    "add_hit_std_deviation",
    "merge_detector_sides",
    "merge_std_deviations",
    "optimize_memory",
    "set_indexes",
]
