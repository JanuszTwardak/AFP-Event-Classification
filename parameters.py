import os
from time import strftime

# preprocessed_df_path = os.path.join(
#     "output",
# )
should_show = False
should_save = True
root_names = ["336505_afp_newhits"]
trees_dimension = 1
dask_n_workers = 1
dask_threads_per_worker = 6
dask_memory_limit = "4GB"
df_names = root_names
memory_chunk_size = "20 MB"
final_results_path = os.path.join(
    "results_test_big_2",
)
scores_file_name = "scores.csv"
scores_dict_name = "scores"
scores_path = os.path.join(final_results_path, scores_dict_name)
preprocessed_data_path = os.path.join("preprocessed_data")
root_name_suffix = "_afp_newhits.root"
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
