# Financial Misinformation Detection

This is a continuous project on Financial Misinformation Detection (FMD).

## News


📢 *Jan. 2026* New preprint paper "All That Glisters Is Not Gold: A Benchmark for Reference-Free Counterfactual Financial Misinformation Detection" at [arXiv](https://arxiv.org/abs/2601.04160).

📢 *Jan. 2026* New preprint paper "Same Claim, Different Judgment: Benchmarking Scenario-Induced Bias in Multilingual Financial Misinformation Detection" at [arXiv]().

📢 *Jan. 2025* Our FMDLlama paper has been accepted by WWW 2025 as a short paper.

📢 *Jan. 2025* The Financial Misinformation Detection Challenge has successfully wrapped up at COLING 2025. Learn more about the [challenge](https://huggingface.co/spaces/TheFinAI/FMD2025).

📢 *Sep. 2024* New preprint paper related to this work: "FMDLlama: Financial Misinformation Detection based on Large Language Models" at [arXiv](https://www.arxiv.org/abs/2409.16452).



## Work 3: All That Glisters Is Not Gold: A Benchmark for Reference-Free Counterfactual Financial Misinformation Detection

### Datasets

- [Link](https://huggingface.co/datasets/CarolynJiang/RFC-Bench-Dataset)



### Citation
```
@article{jiang2026glistersgoldbenchmarkreferencefree,
  title={All That Glisters Is Not Gold: A Benchmark for Reference-Free Counterfactual Financial Misinformation Detection},
  author={Yuechen Jiang and Zhiwei Liu and Yupeng Cao and Yueru He and Chen Xu and Ziyang Xu and Zhiyang Deng and Prayag Tiwari and Xi Chen and Alejandro Lopez-Lira and Jimin Huang and Junichi Tsujii and Sophia Ananiadou},
  journal={arXiv preprint arXiv:2601.04160},
  year={2024}
}
```


## Work 2: Same Claim, Different Judgment: Benchmarking Scenario-Induced Bias in Multilingual Financial Misinformation Detection




## Work 1: FMD LLama: This work also supported Financial Misinformation Detection ([FMD](https://coling2025fmd.thefin.ai/)) challenge at COLING 2025

[Paper arXiv](https://www.arxiv.org/abs/2409.16452)


### Datasets

- [Link](https://huggingface.co/datasets/lzw1008/COLING25-FMD/)


### Usage

#### Data preprocess

You can follow the *practice_data_preprocess.ipynb* file to get instruction train/val/test data in ./data/practice_data/instruct_data/ path.
The default is an instruction example, change accordingly as need.

#### Convert data format

```python
# train
python src/convert_to_conv_data.py --orig_data ./data/practice_data/instruct_data/FMD_train.json --write_data ./data/practice_data/instruct_data/train.json --dataset_name fmd
# val
python src/convert_to_conv_data.py --orig_data ./data/practice_data/instruct_data/FMD_val.json --write_data ./data/practice_data/instruct_data/val.json --dataset_name fmd
```

The commands above are to convert the data into dialogue data format for LLMs training. 
The current format is used for the LLaMA2 series (i.e. "*Human*": "sentence", "*Assistant*": "sentence" ). 
If you need to switch to other LLMs, please make the corresponding modifications.

#### Fine-tune

```python
bash ./src/run_sft.sh
```


#### Inference
```python
bash src/run_inference.sh
```

#### Evaluation
Follow the *evaluation.ipynb* file to get F1, rouge, bertscore, and final score.

### License

This project is licensed under [MIT]. Please find more details in the [MIT](LICENSE) file.

### Citation

```
@inproceedings{liu2025fmdllama,
  title={Fmdllama: Financial misinformation detection based on large language models},
  author={Liu, Zhiwei and Zhang, Xin and Yang, Kailai and Xie, Qianqian and Huang, Jimin and Ananiadou, Sophia},
  booktitle={Companion Proceedings of the ACM on Web Conference 2025},
  pages={1153--1157},
  year={2025}
}
```
