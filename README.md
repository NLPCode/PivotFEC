# README
This repository contains the implementation of the EMNLP 2023 paper: 
"[**PivotFEC: Enhancing Few-shot Factual Error Correction with a Pivot Task Approach using Large Language Modelss**](https://aclanthology.org/2023.findings-emnlp.667/)".
****
##  Abstract
Factual Error Correction (FEC) aims to rectify false claims by making minimal revisions to align them more accurately with supporting evidence. However, the lack of datasets containing false claims and their corresponding corrections has impeded progress in this field. Existing distantly supervised models typically employ the mask-then-correct paradigm, where a masker identifies problematic spans in false claims, followed by a corrector to predict the masked portions. Unfortunately, accurately identifying errors in claims is challenging, leading to issues like over-erasure and incorrect masking. 
To overcome these challenges, we present PivotFEC, a method that enhances few-shot FEC with a pivot task approach using large language models (LLMs). 
Specifically, we introduce a pivot task called factual error injection, which leverages LLMs (e.g., ChatGPT) to intentionally generate text containing factual errors under few-shot settings; then, the generated text with factual errors can be used to train the FEC corrector. 
Our experiments on a public dataset demonstrate the effectiveness of PivotFEC in two significant ways: Firstly, it improves the widely-adopted SARI metrics by 11.3 compared to the best-performing distantly supervised methods. 
Secondly, it outperforms its few-shot counterpart (i.e., LLMs are directly used to solve FEC) by 7.9 points in SARI, validating the efficacy of our proposed pivot task.
****
## Requirements
python 3.10   
pip install openai  
pip install torch==2.0.0+cu118  
pip install transformers==4.24.0  
pip install evaluate==0.4.0  
pip install tensorboardX==2.6  
pip install spacy==3.6.1  
pip install tqdm  
pip install tabulate   


git clone https://github.com/NVIDIA/apex  
cd apex  
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
****

## Dataset
The the evidence-based FEC dataset (FECDATA) used in our paper can be found at https://github.com/j6mes/acl2021-factual-error-correction. For convinence, we extract the gold evidence and retrieved evidence from FECDATA, and merge them into one file. Our processed data are available at https://drive.google.com/drive/folders/16H7WA8rZ-qlGNNIgylrPQox6qAdQq_qM?usp=sharing.

Before running our code, you should download the data, put them into the root directory of this project.


****
## Train our proposed PivotFEC for factual error correction


* Step 1: Creating synthetic data for factual error correction.

Since we use ChatGPT for data augmentation, we first run the following command to create inputs for ChatGPT.
```bash
cd scripts  
sh create_openai_input.sh
```


After executing this command, a directory named 'augment_input_for_openai' will manifest at the root directory. Subsequently, we will employ it to prompt ChatGPT to introduce factual errors into correct claims by executing the following code:  
Please take note: Prior to running the subsequent code, it is essential to copy your API key (https://platform.openai.com/account/api-keys) and substitute it for 'your API key' in line 11 of './models/run_openai.py'.
```bash
cd scripts  
sh run_openai_augmentation.sh
sh extract_openai_results.sh
cd ..
mkdir openai_augmented_data
mv augment_output_for_openai/extracted/* openai_augmented_data
```

The augmented data are available at: https://drive.google.com/drive/folders/1m3iG3WQx_jtPnmqnlJAsREaes8GFJrY9?usp=sharing

* Step 2: Train the corrector on the augmented data generated by ChatGPT for factual error correction.
```bash
cd scripts 
# revise factual errors with retrieved evidence 
sh seq2seq_t5_augmented_data_retrieved_evidence.sh
# revise factual errors with gold evidence
sh seq2seq_t5_augmented_data_gold_evidence.sh
```

## Train the supervised model for factual error correction

Fully supervised baselines estimate the ceiling performance of factual error correction models, under the assumption that a substantial amount of data is accessible. 
For this purpose, we fine-tune T5-base on FECDATA, where the encoder takes the false claim and corresponding evidence as inputs, while the decoder generates the revised claim. 
```bash
cd scripts  
# revise factual errors with retrieved evidence
sh seq2seq_t5_baseline_retrieval_evidence.sh 
# revise factual errors with gold evidence
sh seq2seq_t5_baseline_gold_evidence.sh 
```

## Try our model with the well-trained checkpoints 
| Model           |  Download link
|----------------------|--------|
| 8-shot PivotFEC (ChatGPT) with retrieved or gold evidence | [\[link\]](https://drive.google.com/file/d/1DYy55UmAoaqeyuw_zlR4uXNFFNwj0KYP/view?usp=sharing)  | 
| Supervised T5-base with retrieved or gold evidence| [\[link\]](https://drive.google.com/file/d/17NQLRj0Y7PraSGlWE4nCpvyQDhkeZvjd/view?usp=sharing)  | 

Download the checkpoints, put them into the root directory, and then extract files from the archive by running the following command.
```bash
tar -xzvf checkpoint_name.tar.gz # replace 'checkpoint_name' with the corresponding checkpoint name.
```



## Citation
If you want to use this code in your research, please cite our [paper](https://aclanthology.org/2023.findings-emnlp.667/):
```bash

@inproceedings{he-etal-2023-pivotfec,
    title = "{P}ivot{FEC}: Enhancing Few-shot Factual Error Correction with a Pivot Task Approach using Large Language Models",
    author = "He, Xingwei  and
      Jin, A-Long  and
      Ma, Jun  and
      Yuan, Yuan  and
      Yiu, Siu Ming",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.667",
    pages = "9960--9976",
}

```
