#!/bin/bash

### Processing function
export PYTHONPATH=$PWD:$PYTHONPATH

search_document() {
    CUDA_VISIBLE_DEVICES=0 python src/search_document.py \
        --config config/src/search_document.yaml \
        data_dir=data/$1/raw \
        input_file="$2"_with_id.jsonl \
        base_model=castorini/ance-dpr-question-multi \
        encode_save_dir=data/$1/retrieved \
        batch_size=64 \
        search_topk=10 \
        output_path="$2".jsonl \
        dry_run=True
}


### Search documents per query

# ODQA datasets
search_document_nq() {
    search_document nq train
    search_document nq valid
}

search_document_popqa() {
    search_document popqa test
}

search_document_triviaqa() {
    search_document triviaqa train
    search_document triviaqa valid
    search_document triviaqa test
}

search_document_webqa() {
    search_document webqa train
    search_document webqa test
}

# Long-form QA datasets
search_document_truthfulqa() {
    search_document truthfulqa valid
}

# Fact Checking datasets
search_document_factkg() {
    search_document factkg train
    search_document factkg valid
    search_document factkg test
}

### Main logic
if [ $# -ne 1 ]; then
    echo "사용법: bash src/scripts/search_document.sh [nq | popqa | triviaqa | webqa | truthfulqa | factkg]"
    exit 1
fi

case "$1" in
    nq)
        search_document_nq
        ;;
    popqa)
        search_document_popqa
        ;;
    triviaqa)
        search_document_triviaqa
        ;;
    webqa)
        search_document_webqa
        ;;
    truthfulqa)
        search_document_truthfulqa
        ;;
    factkg)
        search_document_factkg
        ;;
    all)
        search_document_webqa
        search_document_truthfulqa
        search_document_factkg
        ;;
    *)
        echo "잘못된 인자입니다. [nq | popqa | triviaqa | webqa | truthfulqa | factkg] 중 하나를 입력하세요."
        exit 1
        ;;
esac