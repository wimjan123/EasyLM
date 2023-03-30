gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--zone=${ZONE} \
--worker=all --command='git clone https://github.com/wimjan123/EasyLM.git && cd EasyLM/ && export PYTHONPATH="${PWD}:$PYTHONPATH" && sh ./scripts/tpu_vm_setup.sh'


gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
--zone=${ZONE} \
--worker=all --command='cd EasyLM/ && python3 -m EasyLM.models.gptj.gptj_train --train_dataset.type='huggingface''
