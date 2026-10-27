put_item_in_fridge_and_close_door

gr1_open_microwave

gr1_table_multi_object_no_collision

gr1_turn_stand_mixer_knob

gr1_pick_place


gr1_two_arm_box_cleanup

gr1_two_arm_can_sort_random

gr1_two_arm_coffee

gr1_two_arm_drawer_cleanup

gr1_two_arm_lift_tray

gr1_two_arm_pouring


# ----- ----- ----- ----- ----- ----- ----- ----- #


export TSAK_NAME=gr1_two_arm_box_cleanup

export NUM_GENERATED_DEMOS=100

export DATASET_DIR=datasets

python submodules/IsaacLab/scripts/tools/record_demos.py \
  --device cpu \
  --viz kit \
  --xr \
  --cloudxr_env cloudxrjs \
  --dataset_file datasets/${TSAK_NAME}_recorded.hdf5 \
  --num_demos 5 \
  --num_success_steps 10 \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task ${TSAK_NAME} \
  --embodiment gr1_pink \
  --arena_teleop_device openxr


python submodules/IsaacLab/scripts/tools/replay_demos.py \
  --viz kit \
  --device cpu \
  --enable_cameras \
  --dataset_file datasets/${TSAK_NAME}_recorded.hdf5 \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task ${TSAK_NAME} \
  --embodiment gr1_pink


python submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --device cpu \
  --input_file datasets/${TSAK_NAME}_recorded.hdf5 \
  --output_file datasets/${TSAK_NAME}_annotated.hdf5 \
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
  --input_file datasets/${TSAK_NAME}_annotated.hdf5 \
  --output_file datasets/${TSAK_NAME}_generated_${NUM_GENERATED_DEMOS}.hdf5 \
  --enable_cameras \
  --mimic \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task ${TSAK_NAME} \
  --embodiment gr1_pink
  # --viz kit


python submodules/IsaacLab/scripts/tools/replay_demos.py \
  --viz kit \
  --device cpu \
  --enable_cameras \
  --dataset_file datasets/${TSAK_NAME}_generated_${NUM_GENERATED_DEMOS}.hdf5 \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task ${TSAK_NAME} \
  --embodiment gr1_pink


python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \
  --yaml_file isaaclab_arena_gr00t/lerobot/config/gr1_two_arm_config.yaml \
  --hdf5_name ${TSAK_NAME}_generated_${NUM_GENERATED_DEMOS}.hdf5 \
  --language_instruction "Use both hands to bring the box lid close to the bin."

# ----- ----- ----- ----- ----- ----- ----- ----- #


cd submodules/Isaac-GR00T

CUDA_VISIBLE_DEVICES=3 uv run python gr00t/experiment/launch_finetune.py \
  --dataset-path ../../datasets/${TSAK_NAME}_generated_${NUM_GENERATED_DEMOS}/lerobot \
  --output-dir ../../models/ \
  --modality-config-path ../../isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py \
  --global-batch-size 16 \
  --max-steps 30000 \
  --num-gpus 1 \
  --save-steps 1000 \
  --base-model-path /home/mayuhang/models/GR00T-N1.6-3B \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --dataloader-num-workers 4 \
  --embodiment-tag GR1 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --save-total-limit 2


cd submodules/Isaac-GR00T

CUDA_VISIBLE_DEVICES=3 uv run python gr00t/eval/run_gr00t_server.py \
  --modality-config-path ../../isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py \
  --model-path ../../models/checkpoint-30000 \
  --embodiment-tag GR1 \
  --device cuda --host 127.0.0.1 --port 5555



cd IsaacLab-Arena

export TSAK_NAME=gr1_two_arm_box_cleanup

CUDA_VISIBLE_DEVICES=3 python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml \
  --viz none \
  --remote_host 127.0.0.1 \
  --remote_port 5555 \
  --num_episodes 5 \
  --enable_cameras \
  --record_viewport_video \
  --record_camera_video \
  ${TSAK_NAME} \
  --embodiment gr1_joint 
