This is the default branch which will perform space group downstream task. For other tasks, look additional branches. 

Summary:
In this work, we introduce Matfusion, a pre-trained
multi-modal foundation model tailored for crystallographic appli-
cations. Leveraging techniques such as next token prediction and
a diffusion modelling, Matfusion accurately models multi-modal
crystal structure data. Additionally, We present a dataset named
‘OLCF CBED’, which contains 200,000 multi-modal data files,
comprising crystallographic information files (CIF) as text and
corresponding crystal diffraction patterns as images. The Matfu-
sion model was pre-trained using this OLCF CBED dataset. This
pre-trained model can be then fine-tuned for solving a variety
of crystallographic forward and inverse problems with high
accuracy. This work highlights two significant downstream tasks:
the identification of elements in the chemical formula of a crystal
and the determination of crystal space groups. Furthermore,
our experiments conducted on the Frontier supercomputer at
Oak Ridge National Laboratory, utilizing 4096 GPUs, have
demonstrated the model’s noteworthy scaling capabilities. 