
export TSAK_NAME=gr1_two_arm_box_cleanup
export LANGUAGE_INSTRUCTION="move the box lid onto the box"

export TSAK_NAME=gr1_two_arm_drawer_cleanup
export LANGUAGE_INSTRUCTION="pick the cup and open the drawer, then put the cup in the drawer and close the drawer"

export TSAK_NAME=gr1_two_arm_lift_tray
export LANGUAGE_INSTRUCTION="put the two objects in the tray and then lift the tray"

export TSAK_NAME=gr1_two_arm_can_sort_random
export LANGUAGE_INSTRUCTION="pick the can and put it in the box of the same color"

export TSAK_NAME=gr1_two_arm_coffee
export LANGUAGE_INSTRUCTION="insert the coffee pod into the coffee machine and close the lid"

export TSAK_NAME=gr1_two_arm_pouring
export LANGUAGE_INSTRUCTION="pick up the cup, pour the ball into the bowl, then pick up the bowl and place it on the tray"


export NUM_RECORDED_DEMOS=300

export NUM_GENERATED_DEMOS=2000

export DATASET_DIR=datasets


python submodules/IsaacLab/scripts/tools/record_demos.py \
  --device cpu \
  --viz kit \
  --xr \
  --cloudxr_env cloudxrjs \
  --dataset_file datasets/${TSAK_NAME}_recorded_${NUM_RECORDED_DEMOS}.hdf5 \
  --num_demos ${NUM_RECORDED_DEMOS} \
  --num_success_steps 10 \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task ${TSAK_NAME} \
  --embodiment gr1_pink \
  --arena_teleop_device openxr
  # --no-randomize_object_poses
  # --enable_cameras


python submodules/IsaacLab/scripts/tools/replay_record_demos.py \
  --device cpu \
  --enable_cameras \
  --dataset_file datasets/${TSAK_NAME}_recorded_${NUM_RECORDED_DEMOS}.hdf5 \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task ${TSAK_NAME} \
  --embodiment gr1_pink \
  --validate_success_rate
  # --viz kit 


# The GR1 two-arm configs below require four views: head, left wrist, right wrist, and front.
# For older demos, replay with --enable_cameras and ENABLE_FRONT_CAM=True, then reconvert.
# Use only four-view LeRobot datasets and matching four-view checkpoints with these configs.
python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
  --yaml_file isaaclab_arena_gr00t/lerobot/config/gr1_two_arm_config.yaml \
  --hdf5_name ${TSAK_NAME}_recorded_${NUM_RECORDED_DEMOS}_replay.hdf5 \
  --language_instruction "${LANGUAGE_INSTRUCTION}"
  # --skip_failed_episodes


python3 assets/filter_episodes_by_videos.py \
  datasets/${TSAK_NAME}_recorded_${NUM_RECORDED_DEMOS}_replay/lerobot/


python assets/clean_lerobot_dataset.py \
  --dataset datasets/${TSAK_NAME}_recorded_${NUM_RECORDED_DEMOS}_replay/lerobot \
  --apply \
  --original-count ${NUM_RECORDED_DEMOS}


python submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --device cpu \
  --input_file datasets/${TSAK_NAME}_recorded_${NUM_RECORDED_DEMOS}.hdf5 \
  --output_file datasets/${TSAK_NAME}_annotated_${NUM_RECORDED_DEMOS}.hdf5 \
  --mimic \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task ${TSAK_NAME} \
  --embodiment gr1_pink \
  --auto
  # --viz kit


python submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cpu \
  --generation_num_trials ${NUM_GENERATED_DEMOS} \
  --num_envs 20 \
  --input_file datasets/${TSAK_NAME}_annotated_${NUM_RECORDED_DEMOS}.hdf5 \
  --output_file datasets/${TSAK_NAME}_generated_${NUM_GENERATED_DEMOS}.hdf5 \
  --enable_cameras \
  --mimic \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task ${TSAK_NAME} \
  --embodiment gr1_pink
  # --viz kit


python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
  --yaml_file isaaclab_arena_gr00t/lerobot/config/gr1_two_arm_config.yaml \
  --hdf5_name ${TSAK_NAME}_generated_${NUM_GENERATED_DEMOS}.hdf5 \
  --language_instruction "${LANGUAGE_INSTRUCTION}"


python assets/merge_lerobot_datasets.py \
  --datasets datasets/${TSAK_NAME}_recorded_${NUM_RECORDED_DEMOS}_replay/lerobot datasets/${TSAK_NAME}_generated_${NUM_GENERATED_DEMOS}/lerobot \
  --output datasets/${TSAK_NAME}_merged/lerobot \
  --apply


# ----- ----- ----- ----- ----- ----- ----- ----- #


cd submodules/Isaac-GR00T

CUDA_VISIBLE_DEVICES=3 uv run python gr00t/experiment/launch_finetune.py \
  --dataset-path ../../datasets/${TSAK_NAME}_recorded_${NUM_RECORDED_DEMOS}_replay/lerobot \
  --output-dir ../../models/${TSAK_NAME}_${NUM_RECORDED_DEMOS}/ \
  --modality-config-path ../../isaaclab_arena_gr00t/embodiments/gr1/gr1_two_arm_data_config.py \
  --global-batch-size 16 \
  --gradient-accumulation-steps 8 \
  --max-steps 50000 \
  --num-gpus 1 \
  --save-steps 10000 \
  --base-model-path /home/mayuhang/models/GR00T-N1.6-3B \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --dataloader-num-workers 4 \
  --embodiment-tag GR1 \
  --save-total-limit 3
  #--color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \


CUDA_VISIBLE_DEVICES=3 uv run python gr00t/experiment/launch_finetune.py \
  --dataset-path ../../datasets/${TSAK_NAME}_generated_${NUM_GENERATED_DEMOS}/lerobot \
  --output-dir ../../models/${TSAK_NAME}_${NUM_GENERATED_DEMOS}/ \
  --modality-config-path ../../isaaclab_arena_gr00t/embodiments/gr1/gr1_two_arm_data_config.py \
  --global-batch-size 16 \
  --gradient-accumulation-steps 8 \
  --max-steps 50000 \
  --num-gpus 1 \
  --save-steps 10000 \
  --base-model-path /home/mayuhang/models/GR00T-N1.6-3B \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --dataloader-num-workers 4 \
  --embodiment-tag GR1 \
  --save-total-limit 3
  #--color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \


# ----- Multi-task fine-tuning: one model, multiple task datasets ----- #
# This file is a collection of command snippets, not an end-to-end shell script.
# Run the following block from the IsaacLab-Arena repository root.
# Use the GR00T checkout/environment that already runs single-task fine-tuning.
# Default: Arena's submodule. To use an external checkout instead, first set:
# export GROOT_ROOT=/absolute/path/to/Isaac-GR00T-e29d8f
# Keep each task's own language instruction when converting its dataset.
# The same-task merge tool above must NOT be used to merge different tasks.
(
  ARENA_ROOT="$(pwd)"
  GROOT_ROOT="${GROOT_ROOT:-${ARENA_ROOT}/submodules/Isaac-GR00T}"
  cd "${GROOT_ROOT}" || exit 1

  # One weight per path. 1 1 1 gives equal target task sampling weights,
  # regardless of dataset lengths. For a 2:1:1 mixture, use 2 1 1 instead.
  # Each path may also point to that task's generated or same-task merged dataset.
  # Add --dry-run to preview options/mixture without starting training.
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH="${GROOT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    uv run python "${ARENA_ROOT}/assets/launch_finetune_multitask.py" \
    --dataset-paths \
      "${ARENA_ROOT}/datasets/gr1_two_arm_box_cleanup_recorded_300_replay/lerobot" \
      "${ARENA_ROOT}/datasets/gr1_two_arm_drawer_cleanup_recorded_300_replay/lerobot" \
      "${ARENA_ROOT}/datasets/gr1_two_arm_coffee_recorded_300_replay/lerobot" \
    --dataset-weights 1 1 1 \
    --output-dir "${ARENA_ROOT}/models/gr1_two_arm_multitask/" \
    --modality-config-path "${ARENA_ROOT}/isaaclab_arena_gr00t/embodiments/gr1/gr1_two_arm_data_config.py" \
    --global-batch-size 16 \
    --gradient-accumulation-steps 8 \
    --max-steps 50000 \
    --num-gpus 1 \
    --save-steps 10000 \
    --base-model-path /home/mayuhang/models/GR00T-N1.6-3B \
    --no-tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --dataloader-num-workers 4 \
    --embodiment-tag GR1 \
    --save-total-limit 3
)


cd submodules/Isaac-GR00T

CUDA_VISIBLE_DEVICES=3 uv run python gr00t/eval/run_gr00t_server.py \
  --modality-config-path ../../isaaclab_arena_gr00t/embodiments/gr1/gr1_two_arm_data_config.py \
  --model-path ../../models/checkpoint-50000 \
  --embodiment-tag GR1 \
  --device cuda --host 127.0.0.1 --port 5555


cd IsaacLab-Arena

# checkpoint config.json , num_inference_timesteps
# gr1_two_arm_config.yaml , action_chunk_length

CUDA_VISIBLE_DEVICES=3 python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
  --language_instruction "${LANGUAGE_INSTRUCTION}" \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_two_arm_config.yaml \
  --viz none \
  --remote_host 127.0.0.1 \
  --remote_port 5555 \
  --num_episodes 20 \
  --enable_cameras \
  --record_viewport_video \
  --record_camera_video \
  --record_actions \
  --no-finger_action_smoothing \
  ${TSAK_NAME} \
  --embodiment gr1_joint
  # --no-randomize_object_poses


CUDA_VISIBLE_DEVICES=3 python assets/policy_runner_hdf5.py \
  --dataset_file datasets/${TSAK_NAME}_recorded_${NUM_RECORDED_DEMOS}.hdf5 \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
  --language_instruction "${LANGUAGE_INSTRUCTION}" \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_two_arm_config.yaml \
  --viz none \
  --remote_host 127.0.0.1 \
  --remote_port 5555 \
  --num_episodes ${NUM_RECORDED_DEMOS} \
  --enable_cameras \
  --record_viewport_video \
  --record_camera_video \
  ${TSAK_NAME} \
  --embodiment gr1_joint


cd submodules/Isaac-GR00T

CUDA_VISIBLE_DEVICES=3 python gr00t/eval/open_loop_eval.py \
  --dataset-path ../../datasets/${TSAK_NAME}_generated_${NUM_GENERATED_DEMOS}/lerobot \
  --model-path ../../models/${TSAK_NAME}/checkpoint-50000 \
  --embodiment-tag GR1 \
  --traj-ids 0 1 2 3 4 \
  --steps 300 \
  --action-horizon 16


tensorboard --logdir  ./models --host 0.0.0.0 --port 1024


# ----- ----- ----- ----- ----- ----- ----- ----- #

put_item_in_fridge_and_close_door

gr1_open_microwave

gr1_table_multi_object_no_collision

gr1_turn_stand_mixer_knob

gr1_pick_place


gr1_two_arm_box_cleanup, "move the box lid onto the box"

gr1_two_arm_drawer_cleanup, "pick the cup and open the drawer, then put the cup in the drawer and close the drawer"

gr1_two_arm_lift_tray, "put the two objects in the tray and then lift the tray"

gr1_two_arm_can_sort_random, "pick the can and put it in the box of the same color"

gr1_two_arm_coffee, "insert the coffee pod into the coffee machine and close the lid"

gr1_two_arm_pouring, "pick up the cup, pour the ball into the bowl, then pick up the bowl and place it on the tray"

# ----- ----- ----- ----- ----- ----- ----- ----- #
