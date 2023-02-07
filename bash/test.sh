TASK="np"
TEST_PATH=./data/${TASK}/triples_processed/*/test.jsonl
MODELNAME=("PubMedBERT-full" "BioBERT")
MODELTYPE=("BERT" "BERT")
MODEL=("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" "dmis-lab/biobert-base-cased-v1.2")
PROMPT_DIR_PATH=./data/${TASK}/prompts

for i in "${!MODEL[@]}"
do
    echo "MODEL NAME = ${MODELNAME[i]}"
    echo "MODEL TYPE = ${MODELTYPE[i]}"
    echo "MODEL CODE = ${MODEL[i]}"

    for PROMPT_PATH in "${PROMPT_DIR_PATH}"/*
    do
        echo "MANUAL PROMPT: ${PROMPT_PATH}"
        PROMPTFILE=$(basename $PROMPT_PATH)
        PROMPTNAME="${PROMPTFILE%.*}"

        echo "-- compute pronpt analysis"
        python ./BioLAMA/run_manual.py \
            --model_name_or_path ${MODEL[i]} \
            --prompt_path ${PROMPT_PATH} \
            --test_path "${TEST_PATH}" \
            --init_method independent \
            --iter_method confidence \
            --num_mask 3 \
            --max_iter 10 \
            --beam_size 10 \
            --batch_size 16 \
            --output_dir ./output2/${TASK}/${MODELNAME[i]}/${PROMPTNAME} > ./output/${TASK}/${MODELNAME[i]}/${PROMPTNAME}/log.log

        echo "-- compute pronpt bias"
        python ./BioLAMA/run_manual.py \
            --model_name_or_path ${MODEL[i]} \
            --prompt_path ${PROMPT_PATH} \
            --test_path "./data/${TASK}/triples_processed/*/${MODELTYPE[i]}_masked.jsonl" \
            --init_method order \
            --iter_method confidence \
            --num_mask 10 \
            --max_iter 10 \
            --beam_size 5 \
            --batch_size 16 \
            --output_dir ./output2/${TASK}/${MODELNAME[i]}/${PROMPTNAME}/MASKED > ./output/${TASK}/${MODELNAME[i]}/${PROMPTNAME}/MASKED/log.log

    done
done