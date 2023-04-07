export TPU_NAME=gpt-4
export ZONE=us-central2-b
export RUNTIME_VERSION=tpu-vm-v4-base
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0


gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--zone=${ZONE} \
--worker=all --command='git clone https://github.com/wimjan123/EasyLM.git && cd EasyLM/ && export PYTHONPATH=${PWD}:$PYTHONPATH && sh ./scripts/tpu_vm_setup.sh'


gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--zone=${ZONE} \
--worker=all --command='cd EasyLM/ && python3 -m EasyLM.models.gptj.gptj_train --train_dataset.type='huggingface''

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--zone=${ZONE} \
--worker=all --command='cd EasyLM/ && python3 -m EasyLM.models.llama.llama_train --train_dataset.type='huggingface''

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--zone=${ZONE} \
--worker=all --command='kill -9 21200'

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--zone=${ZONE} \
--worker=all \
--command=' cd EasyLM/ && git pull'




gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--zone=${ZONE} \
--worker=all \
--command='cd EasyLM/ && python3 -m EasyLM.models.gptj.gptj_train --train_dataset.type='huggingface' --logger.online True --logger.gcs_output_dir gs://gpt-j-train/GPT-4-JAX'




python3 -m EasyLM.models.gptj.gptj_train --train_dataset.type='huggingface' --logger.online True --logger.gcs_output_dir gs://gpt-j-train/GPT-4-JAX --optimizer.type=adamw --optimizer.accumulate_gradient_steps=32

python3 -m EasyLM.models.llama.llama_train --train_dataset.type='huggingface' --logger.online True --logger.gcs_output_dir gs://gpt-j-train/llama4-output/ --optimizer.type=adamw --optimizer.accumulate_gradient_steps=32

python3 -m EasyLM.models.llama.llama_train --train_dataset.type='huggingface' --logger.online True --logger.gcs_output_dir gs://gpt-j-train/llama4-output/ --optimizer.type=adamw --optimizer.accumulate_gradient_steps=32


python3 -m EasyLM.models.llama.llama_train \
--mp_mesh_dim='16,1' \
--total_steps='2500000' \
--load_llama_config='13b' \
--checkpointer.float_dtype='fp32' \
--load_checkpoint='params::gs://gpt-j-train/llama_stream/13B/streaming_params' \
--initialize_jax_distributed='True' \
--save_model_freq='500' \
--save_milestone_freq='10000' \
--log_freq='500' \
--optimizer.type=adamw \
--optimizer.adamw_optimizer.lr=1e-4 \
--optimizer.accumulate_gradient_steps=32 \
--eval_steps='0' \
--logger.gcs_output_dir='gs://gpt-j-train/GPT-4/' \
--logger.online='True' \
--tokenizer.vocab_file='/llama/tokenizer.model' \
--log_all_worker='True'


python -m EasyLM.scripts.convert_checkpoint \
    --load_checkpoint='params::path/to/checkpoint' \
    --output_file='path/to/output/checkpoint' \
    --streaming=False


python3 -m EasyLM.scripts.convert_checkpoint \
    --load_checkpoint='params::gs://gpt-j-train/llama_stream/13B/streaming_params' \
    --output_file='gs://gpt-j-train/GPT-4/36db7e39151446a0be9539dac57f4833/flax-model.msgpack' \
    --float_dtype='fp32' \
    --streaming=False





--tokenizer.vocab_file='/llama/tokenizer.model'trainstate::

python3 -m EasyLM.scripts.convert_checkpoint \
    --load_checkpoint='params::/convert/gpt-j/train_state_48000' \
    --output_file='/convert/gpt-j/flax/inference-reddit' \
    --streaming=False


python3 -m EasyLM.scripts.convert_checkpoint \
--load_checkpoint='params::/convert/gpt-j/streaming_params_48000' \
--output_file='/convert/gpt-j/flax/4chan-inference' \
--streaming=False



python3 -m EasyLM.models.llama.llama_serve \
    --mp_mesh_dim='-1,1' \
    --load_llama_config='13b' \
    --load_checkpoint='params::gs://gpt-j-train/GPT-4/c3f79936779d4de3a88e73fd32140931/streaming_params' \
    --dtype='bf16' \
    --lm_server.host='0.0.0.0' \
    --lm_server.pre_compile='loglikelihood' \
    --input_length=1024 \
    --seq_length=2048 \
    --tokenizer.vocab_file='/llama/tokenizer.model' 

 10000
110000


python3 -m EasyLM.models.llama.llama_train \
--mp_mesh_dim='16,1' \
--total_steps='110000' \
--load_llama_config='13b' \
--checkpointer.float_dtype='fp32' \
--load_checkpoint='params::gs://gpt-j-train/llama_stream/13B/streaming_params' \
--initialize_jax_distributed='True' \
--save_model_freq='5000' \
--save_milestone_freq='10000' \
--log_freq='500' \
--optimizer.type=adamw \
--optimizer.adamw_optimizer.lr=5e-5 \
--optimizer.adamw_optimizer.end_lr=1e-5 \
--optimizer.adamw_optimizer.lr_warmup_steps='10000' \
--optimizer.adamw_optimizer.lr_decay_steps='100000' \
--optimizer.accumulate_gradient_steps=32 \
--optimizer.bf16_accumulate_gradient='True' \
--eval_steps='0' \
--logger.gcs_output_dir='gs://gpt-j-train/GPT-4/' \
--logger.online='True' \
--tokenizer.vocab_file='/llama/tokenizer.model' \
--log_all_worker='True'