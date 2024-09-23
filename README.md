The final project comprises three individual projects:

1. Comparison of CNN Architectures on Different Datasets.
1. Sequence-to-Sequence Modelling with (and without) Attention Mechanism.
1. Multifunctional NLP and Image Generation Tool using Hugging Face Models.

COMPARISON OF CNN ARCHITECTURES ON DIFFERENT DATASETS

**Problem Statement**:

The project aims to analyse the performance of different CNN architectures on different datasets. The chosen datasets are MNIST, FMNIST, and CIFAR-10. MNIST is a dataset consisting of 60,000 training and 10,000 testing images of handwritten digits. Each image is 28x28 pixels in grayscale. FMNIST (FashionMNIST) is a dataset consisting of 60,000 training images and 10,000 testing images of fashion products. Each image is 28x28 pixels in grayscale. CIFAR-10 is a dataset consisting of 60,000 32x32 images in 10 classes, with 50,000 training and 10,000 testing images. MNIST and FMNIST have grayscale images which test the model’s ability to recognize basic images whereas CIFAR-10 has color images adding complexity for the model to handle detailed and varied data. The chosen CNN networks are LeNet-5, AlexNet, GoogLeNet, VGGNet, ResNet, Xception, and SENet. The comparison will be based on metrics such as loss curves, accuracy, precision, recall, and F1-score.

**Domain**:

Machine Learning, Deep Learning, Computer Vision.

**Approach**:

1\. Load and preprocess the datasets (MNIST, FMNIST, CIFAR-10).

2\. Implement the following CNN architectures: LeNet-5, AlexNet, GoogLeNet, VGGNet, ResNet, Xception, and SENet.

3\. Train each model on each dataset, recording the loss and accuracy metrics.

4\. Evaluate the performance of each model on the test sets using accuracy, precision, recall, and F1-score.

5\. Plot the loss curves and other performance metrics for comparison.

**Technologies used**:

Python - PyTorch

**Packages imported**:

1. torch, torchvision – deep learning
1. from torchvision import datasets, transforms, 

   from torch.utils.data import DataLoader – For datasets

1. torch.nn – neural networks
1. torch.nn.functional – implementing non-linear functions
1. from torch.optim.lr\_scheduler import StepLR, 

   import torch.optim as optim  - for optimizers

1. numpy – numerical python 
1. matplotlib.pyplot – Plotting
1. from sklearn.preprocessing import label\_binarize – convert labels to binary matrix 

**Code to be referred**: CNN\_Assgn.ipynb

**Overview of the code**:

First, I have loaded the MNIST, FMNIST and CIFAR-10 datasets. Then, I have defined three functions: CNN\_par – for defining the required optimizers and loss functions, train – to train the model and evaluate – to test the model. First the neural networks are defined, and the train function is executed for which the loss curves are plotted. Next, the evaluate function is executed for which the model performance is tested through accuracy, precision, recall, F1-score, ROC and AUC curve.

SEQUENCE-TO-SEQUENCE MODELLING WITH (AND WITHOUT) ATTENTION MECHANISM

**Problem Statement**:

The goal of this project is to implement and evaluate sequence-to-sequence (seq2seq) models with attention mechanism. We will train the models on a synthetic dataset where the target sequence is the reverse of the source sequence. The project aims to demonstrate the effectiveness of the attention mechanism in improving seq2seq model performance. The model is evaluated using accuracy (but I have considered precision too) and loss curves during training. 

**Domain**:

Machine Learning, Deep Learning, Natural Language Processing

**Approach**:

1\. Generate a synthetic dataset where each source sequence is a random sequence of integers, and each target sequence is the reverse of the source sequence.

2\. Implement the sequence-to-sequence model with attention mechanism in PyTorch.

3\. Train the model on the synthetic dataset.

4\. Evaluate the model performance using metrics such as loss and accuracy.

5\. Plot the loss curves and other performance metrics for analysis.

**Technologies Used**:

Python – PyTorch.


**Packages Imported**:

1. torch, torch.nn , torch.optim
1. random – generating a random set of integers
1. numpy
1. from torch.utils.data import Dataset, DataLoader
1. matplotlib.pyplot
1. from sklearn.metrics import accuracy\_score, precision\_score – Accuracy and Precision

**Code to be referred**: Seq\_2\_seq\_.ipynb

**Overview of the code**:

First, I have defined a function called generate\_data to generate the source and target of the sequence. Then, a class Seq2SeqDataset is defined to get the synthetic datasets for training and testing respectively. Then, three classes are defined: Attention (for the attention mechanism), Encoder (for encoding the sequence), and Decoder (for decoding the output). The Seq2Seq class is defined with attention where the encoder and decoder are given as inputs. The difference in the class Seq2Seq without attention is that the Encoder and Decoder classes are defined where the Attention class is not defined, and the Decoder doesn’t have an ‘Attention’ part. First, we have considered a small vocabulary size with a small sequence length and have done the training and testing using training and testing respectively. I observed that both Seq2Seq with and without attention worked well. But when I considered, a large vocabulary size with a large sequence length, I observed that Seq2Seq with attention performed well whereas I couldn’t train the model well with 100 epochs in case of without attention! This shows that the attention mechanism improves seq2seq model performance.

MULTIFUNCTIONAL NLP AND IMAGE GENERATION TOOL USING HUGGING FACE MODELS

**Problem Statement**:

The goal of this project is to create a multifunctional tool that allows users to select and utilize different pretrained models from Hugging Face for various tasks. The tool will support text summarization, next word prediction, story prediction, chatbot, sentiment analysis, question answering, and image generation. The front end will provide a user-friendly interface to select the task and input the required text or image for processing. Streamlit app has been utilized for the front end.


**Domain**:

`   `Machine Learning, Deep Learning, Natural Language Processing, Computer Vision

**Approach**:

1\. Set up the environment and install necessary libraries, including Hugging Face Transformers.

2\. Implement a user-friendly front end for task selection and input.

3\. Load and integrate pretrained models from Hugging Face for the following tasks:

`   `- Text Summarization

`   `- Next Word Prediction

`   `- Story Prediction

`   `- Chatbot

`   `- Sentiment Analysis

`   `- Question Answering

`   `- Image Generation

4\. Implement the backend logic to process user inputs and generate outputs using the selected models.

5\. Test the application with various inputs and refine the user interface and backend logic.

**Technologies Used**:

Python – PyTorch

**Packages imported**:

1. torch
1. transformers – to implement the Transformer architecture
1. from transformers import BartForConditionalGeneration, BartTokenizer – Text Summarization
1. from transformers import GPT2LMHeadModel, GPT2Tokenizer – Next word Prediction
1. from transformers import pipeline – Story prediction and Sentiment Analysis
1. from transformers import AutoModelForCausalLM, AutoTokenizer – Chatbot
1. from transformers import AutoTokenizer, AutoModelForQuestionAnswering – Question Answering 
1. diffusers – generative algorithm for ‘*diffusion-based generative models*’.
1. from diffusers import StableDiffusionPipeline – Image Generation
1. streamlit – Streamlit App
1. re – Regular Expressions
1. matplotlib.pyplot 
1. !npm install localtunnel – To create a tunnel website to access Streamlit
1. urllib – to get the password for the tunnel website to access Streamlit 

**Code to be referred**: Hugging\_Face\_Models.ipynb

**Overview of the code**:

I have implemented the following packages as given in the section ‘Packages Imported’. For the streamlit part, I need to have a ‘.py’ file extension. So, I have utilised a single block of and have utilised the command ‘%%writefile Hugging\_face.py’ to save it as a ‘.py’ file. Next in this block, I have defined options for each task and have given the corresponding code part to be executed if that task has been selected. To run the Streamlit app using Google Colab, one must use the command ‘!npm install localtunnel’ – To create a tunnel website. Then a command is used to generate the password for the local tunnel as given in block 38. Then the ‘!streamlit run /content/Hugging\_face.py &>/content/logs.txt &’ command is executed. After that, we command ‘!npx localtunnel --port 8501’ is executed which opens a tunnel website. After entering the password and clicking submit, this will direct to the Streamlit App. On the app, there is a sidebar for choosing the tasks, for which the user can enter the input in the given area and get the desired output.






















