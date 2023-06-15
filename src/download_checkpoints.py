from huggingface_hub import snapshot_download


# repo_id = "naver-clova-ix/donut-base"
# downloaded = snapshot_download(
#     repo_id,
#     revision="official",
#     # allow_patterns=f"main/*",
#     cache_dir="./",
#     # local_dir_use_symlinks=False
# )



# repo_id = "hyunwoongko/asian-bart-ecjk"
# downloaded = snapshot_download(
#     repo_id,
#     cache_dir="./",
# )


# IDEA-CCNL/Yuyuan-GPT2-110M-SciFi-Chinese
repo_id = "IDEA-CCNL/Wenzhong-GPT2-110M"
downloaded = snapshot_download(
    repo_id,
    cache_dir="./",
)