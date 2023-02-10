#!/bin/bash
# WARNING: You MUST use bash to prevent errors

## Sun's Grid Engine parameters
#$ -N REPLACEBYJOBNAME
#$ -S /bin/bash
#$ -M mdelmas@idiap.ch
#$ -P abroad
#$ -cwd

## Environment
# WARNING: your ~/.bashrc will NOT be loaded by SGE
# WARNING: do NOT load your default ~/.bashrc environment blindly; it will most likely break SGE!
# WARNING: include ONLY the SETSHELLs required for the job at hand; some SETSHELLs will break SGE!
# ... SETSHELL

. /idiap/resource/software/initfiles/shrc
SETSHELL grid


export TRANSFORMERS_CACHE="/idiap/temp/mdelmas/.cache/huggingface/hub"
export PATH="/idiap/temp/mdelmas/miniconda3/bin:$PATH"

source activate prompting

## JOB
TASK="np"
BATCHSIZE=64
DIR="/idiap/temp/mdelmas/ABRoad/app/Exp-Taxa-NP"
PROMPT_TOKEN_LEN=20

MODELNAME=("ChemicalBERT" "BioBERT" "PubMedBERT" "PubMedBERT-full")
MODELTYPE=("BERT" "BERT" "BERT" "BERT")
INITMETHODS=("order" "confidence")
ITERMETHODS=("none" "confidence")
MODEL=("recobo/chemical-bert-uncased" "dmis-lab/biobert-base-cased-v1.2" "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext") #  "

PROMPTS=("${DIR}/data/${TASK}/prompts/manual1.jsonl" "${DIR}/data/${TASK}/prompts/manual2.jsonl" "FS") 

TRAIN_PATH=${DIR}/data/${TASK}/triples_processed/*/train.jsonl
DEV_PATH=${DIR}/data/${TASK}/triples_processed/*/dev.jsonl
TEST_PATH=${DIR}/data/${TASK}/triples_processed/*/test.jsonl

# The part taht should be completed before sending the job
i=REPLACEBYMODELINDEX
j=REPLACEBYPROMPATH

PROMPT=${PROMPTS[$j]}

for INIT in "${INITMETHODS[@]}"
do

echo "TEST WITH INIT METHOD = ${INIT}"

for ITER in "${ITERMETHODS[@]}"
do
    echo "TEST WITH ITER METHOD = ${ITER}"
    
    if [ $PROMPT = "FS" ]
    then
        echo "-- compute pronpt analysis"
        # mkdir -p ${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPT}/${INIT}/${ITER}
        mkdir -p ${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPT}/${INIT}

        python ${DIR}/BioLAMA/run_optiprompt.py \
            --model_name_or_path "${MODEL[i]}" \
            --train_path "${TRAIN_PATH}" \
            --dev_path "${DEV_PATH}" \
            --test_path "${TEST_PATH}" \
            --num_mask 10 \
            --init_method ${INIT} \
            --iter_method ${ITER} \
            --max_iter 10 \
            --beam_size 5 \
            --batch_size ${BATCHSIZE} \
            --lr 3e-3 \
            --epochs 10 \
            --seed 0 \
            --draft \
            --prompt_token_len ${PROMPT_TOKEN_LEN} \
            --output_dir "${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPT}/${INIT}/${ITER}"

        echo "-- compute pronpt bias"
        mkdir -p ${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPT}/${INIT}/${ITER}_draft/MASKED
        
        python ${DIR}/BioLAMA/run_optiprompt.py \
            --model_name_or_path "${MODEL[i]}" \
            --train_path "${TRAIN_PATH}" \
            --dev_path "${DEV_PATH}" \
            --test_path "${DIR}/data/${TASK}/triples_processed/*/${MODELTYPE[i]}_masked.jsonl" \
            --prompt_vector_dir "${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPT}/${INIT}/${ITER}_draft" \
            --num_mask 10 \
            --init_method ${INIT} \
            --iter_method ${ITER} \
            --max_iter 10 \
            --beam_size 5 \
            --batch_size ${BATCHSIZE} \
            --seed 0 \
            --draft \
            --prompt_token_len ${PROMPT_TOKEN_LEN} \
            --output_dir "${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPT}/${INIT}/${ITER}_draft/MASKED"
    else
        PROMPTFILE=$(basename $PROMPT)
        PROMPTNAME="${PROMPTFILE%.*}"
        echo "-- compute pronpt analysis"
        # mkdir -p ${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER}
        mkdir -p ${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPTNAME}/${INIT}

        python ${DIR}/BioLAMA/run_optiprompt.py \
            --model_name_or_path "${MODEL[i]}" \
            --train_path "${TRAIN_PATH}" \
            --dev_path "${DEV_PATH}" \
            --test_path "${TEST_PATH}" \
            --num_mask 10 \
            --init_method ${INIT} \
            --iter_method ${ITER} \
            --max_iter 10 \
            --beam_size 5 \
            --batch_size ${BATCHSIZE} \
            --lr 3e-3 \
            --epochs 10 \
            --seed 0 \
            --draft \
            --init_manual_template \
            --prompt_path "${PROMPT}" \
            --output_dir "${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER}"

        echo "-- compute pronpt bias"
        mkdir -p ${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER}_imt_draft/MASKED
        
        python ${DIR}/BioLAMA/run_optiprompt.py \
            --model_name_or_path "${MODEL[i]}" \
            --train_path "${TRAIN_PATH}" \
            --dev_path "${DEV_PATH}" \
            --test_path "${DIR}/data/${TASK}/triples_processed/*/${MODELTYPE[i]}_masked.jsonl" \
            --prompt_vector_dir "${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER}_imt_draft" \
            --num_mask 10 \
            --init_method ${INIT} \
            --iter_method ${ITER} \
            --max_iter 10 \
            --beam_size 5 \
            --batch_size ${BATCHSIZE} \
            --seed 0 \
            --draft \
            --init_manual_template \
            --prompt_path "${PROMPT}" \
            --output_dir "${DIR}/output/${TASK}/opti/${MODELNAME[i]}/${PROMPTNAME}/${INIT}/${ITER}_imt_draft/MASKED"
    fi
    done
done