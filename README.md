# OPT-GAN: A Broad-Spectrum Global Optimizer for Black-box Problems by Learning Distribution

## Usage
This repository contains the source code of OPT-GAN and examples of how to use it to optimize given black-box problems, such as 'Conformal Bent Cigar' or 'NEEP'.

### Code of OPT-GAN
This folder includes the code for OPT-GAN and an example of how to use it to optimize the 'Conformal Bent Cigar' problem. If you wish to use OPT-GAN for optimizing other black-box problems, you will need to replace the corresponding content with the target problem.
``` python
if func == "Conformal_Bent_Cigar":
    problem = Problem(problem_Conformal_Bent_Cigar) #define the target black-box problem
    upper = 5 # the up bound of the black-box problem's solution
    lower = -5 # the low bound of the black-box problem's solution
```

### Custom Parameter of the OPT-GAN
To specify the parameters of OPT-GAN while optimizing a black-box problem, you can modify them when running the Python file as follows:
```bash
python experiment_Conformal_Bent_Cigar.py --gamma=1.5 --lambda2=0.3 --pop=30 --optset=150 --func_dim=2 --func_ins=0 --func_id=Conformal_Bent_Cigar --func_alg=OPT-GAN --maxfes=50000
```

### Eamples \ example_neep
The folder also contains an example of how to use OPT-GAN to optimize the 'NEEP' problem.

## Cite
If our work is helpful to you, please kindly cite our paper as:
```
@article{Lu_Ning_Liu_Sun_Zhang_Yang_Wang_2023,
    title        = {OPT-GAN: A Broad-Spectrum Global Optimizer for Black-Box Problems by Learning Distribution},
    volume       = {37},
    url          = {https://ojs.aaai.org/index.php/AAAI/article/view/26468},
    doi          = {10.1609/aaai.v37i10.26468},
    number       = {10},
    journal      = {Proceedings of the AAAI Conference on Artificial Intelligence},
    author       = {Lu, Minfang and Ning, Shuai and Liu, Shuangrong and Sun, Fengyang and Zhang, Bo and Yang, Bo and Wang, Lin},
    year         = {2023},
    month        = {Jun.},
    pages        = {12462-12472}
}
```
