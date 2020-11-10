# Multilingual AMR to Text Generation 

Documentation for our paper: https://www.aclweb.org/anthology/2020.emnlp-main.231.pdf

## Training Details: 

We use [Europarl](https://www.statmt.org/europarl/) for training,  which is a corpus of European Parliamentary data. We parse the Europarl English text into AMR with the [JAMR parser](https://github.com/jflanigan/jamr). We subsequently linearize the AMR from a graph into a sequence, following https://arxiv.org/abs/1704.08381 (note: we do not apply anonymization to entities). 

To train our models, we use [fairseq](https://github.com/pytorch/fairseq/). We used SPM on the target side to train our models, which you can download:

```bash
wget https://dl.fbaipublicfiles.com/amr_mtext/m_amr_to_text.spm
```

## Evaluation Details:

For evaluation, we use two evaluation datasets:

(1) Europarl, using JAMR to create the English AMR. Detail: the original Europarl dataset was distributed with "common-test", which was an n-way parallel evaluation dataset. In early WMT competitions, the evaluation dataset was the last quarter of the available previous data. Later, Europarl data was added for additional countries as they joined the European Union, but the common-test set was not revisited. Thus, in our work, we use the common-test set (split into valid and test) where available, and take a portion of the training set when the languages are not available in the common-test set. 

For reproducibility, we provide the English AMR input and corresponding Multilingual output in various languages. The multilingual output contains raw text and SPM versions. 

```bash
wget https://dl.fbaipublicfiles.com/amr_mtext/data.tar.gz
```

(2) [Crosslingual AMR dataset](https://www.aclweb.org/anthology/N18-1104/). For this dataset, please contact Damonte and Cohen. 


## Reproduce Model Results

To evaluate our models, please download fairseq and download our pretrained model:

```bash
wget https://dl.fbaipublicfiles.com/amr_mtext/m_amr_to_text.pt
wget https://dl.fbaipublicfiles.com/amr_mtext/dict.merge_source.txt
wget https://dl.fbaipublicfiles.com/amr_mtext/dict.merge_target.txt
```

Then, to run the evaluation, follow the commands below, assuming you already have fairseq cloned:

```bash
for INPUT_LANG in bg cs da de el es et fi fr hu it lt lv nl pl pt ro sk sl sv; do
    sed -e "s/^/${INPUT_LANG}_token /" test.europarl.$INPUT_LANG-en.en > file.input

    cat file.input | CUDA_VISIBLE_DEVICES=0 python fairseq/interactive.py $data_bin_here --path m_amr_to_text.pt --source-lang merge_source --target-lang merge_target --batch-size 32 --beam 5 --buffer-size 64 >output.txt

    cat output.txt | grep ^H | cut -f3- > hypotheses.txt

    python fairseq/scripts/spm_decode.py --model=m_amr_to_text.spm --input_format=piece --input hypotheses.txt > real_hypoth.txt
    python fairseq/scripts/spm_decode.py --model=m_amr_to_text.spm --input_format=piece --input test.europarl.$INPUT_LANG-en.$INPUT_LANG.spm > real_targets.txt

    echo "Processing Language $INPUT_LANG"
    cat real_hypoth.txt | sacrebleu real_targets.txt
done
```

The output of this evaluation script should be:

```
Processing Language bg
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 35.7 67.2/43.7/31.0/22.4 (BP = 0.944 ratio = 0.946 hyp_len = 129193 ref_len = 136593)
Processing Language cs
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 29.5 61.4/37.1/24.8/17.1 (BP = 0.940 ratio = 0.941 hyp_len = 111194 ref_len = 118124)
Processing Language da
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 22.0 55.7/28.2/17.1/10.8 (BP = 0.948 ratio = 0.949 hyp_len = 125982 ref_len = 132723)
Processing Language de
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 17.5 50.9/22.9/12.7/7.5 (BP = 0.958 ratio = 0.959 hyp_len = 126094 ref_len = 131447)
Processing Language el
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 14.6 47.6/21.7/11.9/6.7 (BP = 0.862 ratio = 0.871 hyp_len = 130434 ref_len = 149735)
Processing Language es
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 25.3 58.6/31.9/20.1/12.9 (BP = 0.955 ratio = 0.956 hyp_len = 137666 ref_len = 143956)
Processing Language et
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 21.2 54.6/28.0/16.6/10.3 (BP = 0.937 ratio = 0.939 hyp_len = 95117 ref_len = 101273)
Processing Language fi
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 13.4 44.8/18.7/9.9/5.6 (BP = 0.910 ratio = 0.914 hyp_len = 92201 ref_len = 100910)
Processing Language fr
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 20.3 54.1/28.1/17.0/10.6 (BP = 0.885 ratio = 0.891 hyp_len = 136604 ref_len = 153291)
Processing Language hu
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 24.7 59.0/32.2/19.7/12.8 (BP = 0.938 ratio = 0.940 hyp_len = 111082 ref_len = 118208)
Processing Language it
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 19.0 51.3/25.4/14.8/8.9 (BP = 0.931 ratio = 0.934 hyp_len = 125710 ref_len = 134640)
Processing Language lt
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 25.5 58.5/32.7/20.9/13.8 (BP = 0.937 ratio = 0.939 hyp_len = 104282 ref_len = 111050)
Processing Language lv
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 27.9 61.4/35.3/23.1/15.9 (BP = 0.936 ratio = 0.938 hyp_len = 107252 ref_len = 114359)
Processing Language nl
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 19.4 53.1/25.0/14.5/8.9 (BP = 0.954 ratio = 0.955 hyp_len = 131278 ref_len = 137446)
Processing Language pl
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 24.4 56.4/31.6/19.9/12.9 (BP = 0.937 ratio = 0.939 hyp_len = 109950 ref_len = 117099)
Processing Language pt
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 22.2 56.5/29.6/18.2/11.5 (BP = 0.914 ratio = 0.917 hyp_len = 133364 ref_len = 145388)
Processing Language ro
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 32.1 63.3/40.0/27.4/19.1 (BP = 0.946 ratio = 0.947 hyp_len = 131383 ref_len = 138741)
Processing Language sk
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 31.4 62.4/39.1/26.8/18.8 (BP = 0.944 ratio = 0.945 hyp_len = 111595 ref_len = 118042)
Processing Language sl
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 33.6 63.9/40.9/28.6/20.5 (BP = 0.954 ratio = 0.955 hyp_len = 114614 ref_len = 120023)
Processing Language sv
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 19.5 53.2/25.3/14.7/8.8 (BP = 0.955 ratio = 0.956 hyp_len = 118938 ref_len = 124394)
```

## Citation

If you found this useful, please consider citing our work:

```bibtex
@inproceedings{fan-gardent-2020-multilingual,
    title = "Multilingual {AMR}-to-Text Generation",
    author = "Fan, Angela  and Gardent, Claire",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    pages = "2889--2901",
}
```

##  License

This repository is MIT-licensed. The license applies to the pre-trained models as well.
