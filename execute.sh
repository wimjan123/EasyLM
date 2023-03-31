export TPU_NAME=gpt-4
export ZONE=us-central2-b
export RUNTIME_VERSION=tpu-vm-v4-base
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0


gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--zone=${ZONE} \
--worker=all --command='git clone https://github.com/wimjan123/EasyLM.git && cd EasyLM/ && export PYTHONPATH="${PWD}:$PYTHONPATH" && sh ./scripts/tpu_vm_setup.sh'


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
--worker=all --command=' cd EasyLM/ && git pull'




gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--zone=${ZONE} \
--worker=all \
--command='cd EasyLM/ && python3 -m EasyLM.models.gptj.gptj_train --train_dataset.type='huggingface' --logger.online True --logger.gcs_output_dir gs://gpt-j-train/GPT-4-JAX'