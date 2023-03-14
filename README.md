# Evaluating the potential of Language Models as Knowledge Bases to support antibiotic discovery

We are very grateful to Mujeen Sung, Jinhyuk Lee, Sean S. Yi, Minji Jeon, Sungdong Kim and Jaewoo Kang for publishing the code of their paper *Can Language Models be Biomedical Knowledge Bases?*, which has greatly inspired this work.

This documentation provides all the steps to reproduce our analysis.
 
## Getting Started - installing conda env

```bash
conda env create -f env/env.yml
```

Then,
```bash
conda activate prompting
```

## Getting data and output experiments

Please, download the *data* and *output* directory from [ZENODO LINK]

* * *

## Discriminant analysis

Can language models distinguish between true and false assertions about relations between fungi and natural product, or, about the antibiotic activity of natural products ?

The dataset is available at: *data/np/discriminant-analysis/dataset.tsv*

### Compute the analysis
Please, use *Analyses/Discriminative-analysis/discriminative-analysis.ipynb* to re-generate the measures and use *Analyses/Discriminative-analysis/discriminant-analysis-figure.Rmd* to regenerate the figures.

For the first step, pre-computed results from the paper are also provided in *output/np/discriminant-analysis*

* * *

## Tokenization analysis

Before evaluating the potential of models to predict fungi related to natural product and vice-versa, we estimated how chemicals and taxa names are decomposed by the native tokenizers of the models.

Data available in: *data/np/pre-processing*

### Compute the analysis

Use *Analyses/tokenization-analysis/tokenization-analysis.ipynb* to reproduce the tables.

* * *

## Prediction analysis

Can pre-trained language models predict the fungi producing a natural product or vice-versa ?

### Get or re-create training, dev and test datasets

* Original dataset available at: *data/np/triples_processed*

* To regenerate a dataset, please use:

```{bash}
TOPN=5000
TOPK=50
python preprocessing/process_abroad_triples.py  \
    --input_pairs="data/np/pre-processing/taxon-np-list.csv"  \
    --input_taxa_names="data/np/pre-processing/ALL-taxons-ids-names.tsv"  \
    --input_chem_main_names="data/np/pre-processing/wikidata/np_names_wikidata.tsv"  \
    --input_chem_syn_names="data/np/pre-processing/wikidata/np_synonyms_wikidata.tsv"  \
    --property="rP703"  \
    --topn=${TOPN} \
    --topk=${TOPK} \
    --outdir="data/np/triples_processed"

python preprocessing/process_abroad_triples.py  \
    --input_pairs="data/np/pre-processing/taxon-np-list.csv"  \
    --input_taxa_names="data/np/pre-processing/ALL-taxons-ids-names.tsv"  \
    --input_chem_main_names="data/np/pre-processing/wikidata/np_names_wikidata.tsv"  \
    --input_chem_syn_names="data/np/pre-processing/wikidata/np_synonyms_wikidata.tsv"  \
    --property="P703" \
    --topn=${TOPN} \
    --topk=${TOPK} \
    --outdir="data/np/triples_processed" \
```

For help of the parameters use:
```{bash}
process_abroad_triples.py --help
```

Info: By default, the datasets are split in 0.4, 0.1, 0.5 for train, dev and test, as in *Sung et al.*. As we want to assess the maximal performances of the models without generalisation purposes, we used 20% of the test set for building the development set and selecting the best checkpoint during training. Then, we merged the initial train and dev set, and, used the *shuf -n X* to extract the new dev set from the test set.

### Get or re-generate the results

All generated experiments are available in *output/manual/* and *output/opti* for manual prompts and OptiPrompt respectively. All combinations of prompts and multi-tokens decoding strategies were evaluated for each selected models.

All experiments can be re-computed following the documentation already provided in the [original repository](https://github.com/dmis-lab/BioLAMA) of *Sung et al.*

### Compute figures

All figures and tables from the paper can be reproduced from *Analyses/prediction-analysis/prediction-analysis-figures-and-tables.Rmd*
Please, consider that the indexing of the manual prompt (*manual1* or *manual2*) is **inversed for P703** in the paper compared to what is indicated in the prompt files at (*data/np/prompts*).

Warning: For practical reasons, when computing the predictions, *manual1* is set to "The fungus [X] produces the compound [Y]." when predicting the chemical and "The compound [X] was isolated and identified from culture of the fungus [Y]." when predicting the fungus. Also, *manual2* is set to "The compound [Y] was isolated and identified from culture of the fungus [X]." when predicting the chemical and "The compound [X] was isolated and identified from culture of the fungus [Y]." when predicting the fungus. However, we reverse the prompt for P703 of manual1 to manual2 and vice-versa when interpreting the results.

