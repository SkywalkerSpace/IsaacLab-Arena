put_item_in_fridge_and_close_door

gr1_open_microwave

gr1_table_multi_object_no_collision

gr1_turn_stand_mixer_knob


cd submodules/Isaac-GR00T

CUDA_VISIBLE_DEVICES=3 uv run python gr00t/experiment/launch_finetune.py \
  --dataset-path ../../datasets/isaaclab_arena/sequential_static_manipulation_tutorial/ranch_bottle_into_fridge_generated_100/lerobot \
  --output-dir ../../models/isaaclab_arena/sequential_static_manipulation_tutorial \
  --modality-config-path ../../isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py \
  --global-batch-size 16 \
  --max-steps 30000 \
  --num-gpus 1 \
  --save-steps 1000 \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --no-tune-llm \
  --tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --dataloader-num-workers 4 \
  --embodiment-tag GR1 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --save-total-limit 52

  # --no-tune-visual \


cd submodules/Isaac-GR00T

CUDA_VISIBLE_DEVICES=2 uv run python gr00t/eval/run_gr00t_server.py \
  --modality-config-path ../../isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py \
  --model-path ../../models/isaaclab_arena/sequential_static_manipulation_tutorial/checkpoint-30000 \
  --embodiment-tag GR1 \
  --device cuda --host 127.0.0.1 --port 5555



cd -

CUDA_VISIBLE_DEVICES=2 python isaaclab_arena/evaluation/policy_runner.py \
  --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
  --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml \
  --viz none \
  --remote_host 127.0.0.1 \
  --remote_port 5555 \
  --num_episodes 5 \
  --enable_cameras \
  --output_base_dir outputs \
  --record_viewport_video \
  --record_camera_video \
  put_item_in_fridge_and_close_door \
  --embodiment gr1_joint \
  --object ranch_dressing_hope_robolab

