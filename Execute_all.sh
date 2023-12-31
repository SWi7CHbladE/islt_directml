#!/bin/bash

echo launching
conda init

echo opening environment
chdir "~/islt_directml/"

conda activate slt_directml
echo environment activated

rm -r result_stats
mkdir result_stats
head_number=8
while [ $head_number -lt 33 ]; do
    batch_number=8
    mkdir result_stats/${head_number}head
    echo head $head_number
    while [ $batch_number -lt 257 ]; do
        echo batch $batch_number
        mkdir result_stats/${head_number}head/${batch_number}batch
        python -m signjoey train configs/${head_number}head/sign_${head_number}head_${batch_number}batch.yaml
        cp sign_sample_model/${head_number}head/${batch_number}batch/train.log result_stats/${head_number}head/${batch_number}batch
        cp sign_sample_model/${head_number}head/${batch_number}batch/validations.txt result_stats/${head_number}head/${batch_number}batch
        cp sign_sample_model/${head_number}head/${batch_number}batch/txt.vocab result_stats/${head_number}head/${batch_number}batch
        cp sign_sample_model/${head_number}head/${batch_number}batch/gls.vocab result_stats/${head_number}head/${batch_number}batch
        mkdir result_stats/${head_number}head/${batch_number}batch/txt
        cp sign_sample_model/${head_number}head/${batch_number}batch/txt/* result_stats/${head_number}head/${batch_number}batch/txt
        cp sign_sample_model/${head_number}head/${batch_number}batch/config.yaml result_stats/${head_number}head/${batch_number}batch
        echo batch $batch_number complete
        ((batch_number = batch_number * 2))
        done
    echo head ${head_number} complete
    ((head_number = head_number * 2))
    done

echo complete
