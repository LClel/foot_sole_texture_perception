# The code and data related to the paper titled "Texture perception at the foot sole"

### Reference
Cleland, L. D., Rupani, M., Blaise, C. R., Elmmers, T. J., & Saal, H. P. (2024). Texture perception at the foot sole: Comparison between walking, sitting, and the hand. Journal of Neurophysiology. DOI: 10.1152/jn.00170.2024.

## Paper authors:
* Luke D Cleland<sup>1,2,3*</sup> - ldcleland1@sheffield.ac.uk - repository manager - ORCID ID: 0000-0001-8486-2780
* Mia Rupani<sup>1</sup>
* Celia Blaise<sup>1,4</sup> - ORCID ID: 0009-0006-9646-7394
* Toby J Ellmers<sup>5</sup> - t.ellmers@imperial.ac.uk - ORDIC ID: 0000-0001-9595-6360
* Hannes P Saal<sup>1,2,3</sup> - h.saal@sheffield.ac.uk - ORCID ID: 0000-0002-7544-0196

<sup>1</sup> Active Touch Laboratory, Department of Psychology, University of Sheffield, Sheffield, UK <br />
<sup>2</sup> Insigneo Institute for in silico Medicine, University of Sheffield, Sheffield, UK <br />
<sup>3</sup> Neuroscience Institute, University of Sheffield, Sheffield, UK <br />
<sup>4</sup> Cognitive Studies, Department of Philosophy, University of Sheffield, Sheffield, UK <br />
<sup>5</sup> Centre for Vestibular Neurology, Department of Brain Sciences, Imperial College London, London, UK <br />
<sup>*</sup> Corresponding author

### Repository author
All code within this repository is authored by Luke Cleland unless otherwise specified.

### Contents of the repository
* `/administration` contains the project checklist with task descriptions, consent form and information sheet
* `/code` contains all files of code required to collate, process and analyse the data
* `/processed_data` contains preprocessed data that is used for data analysis. These files are used for later analysis and figure generations
* `/individual_figures` contains the individual figures used to generate panels within the manuscript
* `/paper_figures` contains the panels found within the manuscript

### Data on the Open Science Framework ()
* `/raw_data` contains all raw data files relating to participant ratings. Should be saved in a folder named `Data` within this repository
     - `/participant id`
             - `/.xslx` raw data files
* `/processed_data` contains files with pre-processed data in will be saved following the processing pipeline. 

## Versions:
* Python 3.8.5
* Numpy 1.19.1
* Pandas 2.0.3
* Matplotlib 3.3.1
* Seaborn 0.11.1
* Scipy 1.10.1
* Scikit-learn - 0.23.2    

## Acknowledgements
L.D.C. is supported by a studentship from the MRC Discovery Medicine North (DiMeN)
Doctoral Training Partnership (MR/N013840/1).
T.J.E. is supported by a Wellcome Trust Sir Henry Wellcome Fellowship (222747/Z/21/Z).
