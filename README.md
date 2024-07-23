# SympTEMIST Evaluation Library

This repository contains the official evaluation library for the [SympTEMIST Shared Task](https://temu.bsc.es/symptemist).
SympTEMIST is a shared task/challenge and set of resources for the detection and normalization of symptoms, signs and findings in medical documents in Spanish.
For more information about the task, data, evaluation metrics, ... please visit the task's website.


## Requirements

To use this scorer, you'll need to have Python 3 installed in your computer. Clone this repository, create a new virtual environment and then install the required packages:

```bash
git clone https://github.com/nlp4bia-bsc/symptemist_evaluation_library
cd symptemist_evaluation_library
python3 -m venv venv/
source venv/bin/activate
pip install -r requirements.txt
```

The SympTEMIST task data is available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.8223653).

## Usage Instructions

This program compares two .TSV files, with one being the reference file (i.e. Gold Standard data provided by the task organizers) and the other being the predictions or results file (i.e. the output of your system). Your .TSV file needs to have the following structure:

- For sub-task 1 (SymptomNER): filename, label, start_span, end_span, text
- For sub-task 2 (SymptomNorm): filename, label, start_span, end_span, text, code
- For sub-task 3 (SymptomMultiNorm): filename, label, start_span, end_span, text, code

Once you have your predictions file in the appropriate format and the reference data ready, you can run the library from your terminal using the following command:

```commandline
python3 symptemist_evaluation.py -r symptemist_task1_ref.tsv -p symptemist_task1_pred.tsv -t ner -o scores/
```

The output will be a .txt file saved in your desired location (`-o` option) with the following filename: symptemist_results_{task}_{timestamp}.txt

These are the possible arguments:

+ ```-r/--reference```: path to Gold Standard TSV file with the annotations
+ ```-p/--prediction```: path to predictions TSV file with the annotations
+ ```-o/--output```: path to save the scoring results file
+ ```-t/--task```: subtask name (```ner```, ```norm```, or ```multi```).
+ ```-v/--verbose```: whether to include the evaluation of every individual document in the scoring results file


### Citation

Please cite the following article if you use the SympTEMIST evaluation library or data:

@inproceedings{symptemist,
  author       = {Lima-L{\'o}pez, Salvador and Farr{\'e}-Maduell, Eul{\`a}lia and Gasco-S{\'a}nchez, Luis and Rodr{\'i}guez-Miret, Jan and Krallinger, Martin},
  title        = {{Overview of SympTEMIST at BioCreative VIII: Corpus, Guidelines and Evaluation of Systems for the Detection and Normalization of Symptoms, Signs and Findings from Text}},
  booktitle    = {Proceedings of the BioCreative VIII Challenge and Workshop: Curation and Evaluation in the era of Generative Models},
  year         = 2023
}

## Contact
If you have any questions or suggestions, please contact us at:

- Salvador Lima-López (<salvador [dot] limalopez [at] gmail [dot] com>)
